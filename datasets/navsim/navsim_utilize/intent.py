"""
Interaction Feature Extractor for NAVSIM
==========================================
Covers all road users EXCEPT pedestrians (handled separately):
  - vehicles     (cars, trucks, buses)
  - bicycles     (cyclists)
  - static objs  (traffic_cone, barrier, czone_sign, generic_object)

Three layers of extraction:
  1. Per-agent temporal tracks  — same approach as pedestrian extractor,
                                   matched by track_token across history frames
  2. Ego-relative kinematic features — position/velocity expressed in ego frame
                                        with explicit ego-centric motion cues
  3. Pairwise interaction state  — for each agent pair (ego, agent_i):
                                    TTC, relative approach rate, heading alignment,
                                    and a soft intent signal from acceleration history

Intent taxonomy (from design doc, 6 classes):
  0  UNCOMMITTED  — approaching but no clear decision yet
  1  ASSERTING    — maintaining/increasing speed toward conflict
  2  YIELDING     — decelerating toward conflict zone
  3  COMMITTED    — already in/past conflict point
  4  UNAWARE      — no speed change despite converging path
  5  CONTESTED    — both agents asserting simultaneously (danger)

Without lane data, we compute soft intent from kinematics alone.
This is the "fallback" mode described in the design doc.

Box layout in annotations.boxes[i]:
  [0] X        ego-frame forward [m]
  [1] Y        ego-frame lateral [m]
  [2] Z        height [m]  (unused)
  [3] LENGTH   along heading [m]
  [4] WIDTH    perpendicular to heading [m]
  [5] HEIGHT   vertical [m]
  [6] HEADING  in ego frame [rad]

velocity_3d[i] = [vx, vy, vz] in ego frame (vz unused)

History: frames[0]=oldest, frames[-1]=current, 2Hz → dt=0.5s between frames
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from navsim.common.dataclasses import AgentInput, Frame

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_DT = 0.5          # seconds between 2Hz history frames
EGO_LENGTH = 4.5          # metres — Pacifica parameters from codebase
EGO_WIDTH  = 2.0

# Agent type strings (from navsim/planning/scenario_builder/navsim_scenario_utils.py)
TYPE_VEHICLE  = "vehicle"
TYPE_BICYCLE  = "bicycle"
TYPE_STATIC   = frozenset({"traffic_cone", "barrier", "czone_sign", "generic_object"})
TYPE_PEDESTRIAN = "pedestrian"   # excluded here, handled by pedestrian extractor

DYNAMIC_TYPES = frozenset({TYPE_VEHICLE, TYPE_BICYCLE})
ALL_NON_PED   = frozenset({TYPE_VEHICLE, TYPE_BICYCLE} | TYPE_STATIC)

# Dimension priors for fallback when box data is noisy
DIMENSION_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "vehicle":        (4.5,  2.0,  1.5),
    "bicycle":        (1.8,  0.6,  1.5),
    "traffic_cone":   (0.5,  0.5,  0.9),
    "barrier":        (2.5,  0.5,  0.8),
    "czone_sign":     (0.5,  0.5,  1.0),
    "generic_object": (1.0,  1.0,  1.0),
}

# Interaction mass: how much planning weight does this agent type demand?
# Derived from stopping distance physics: KE ∝ mass × v², but also collision consequence.
INTERACTION_MASS: Dict[str, float] = {
    "vehicle":        10.0,   # ~1500kg car → heavy, hard to stop, high collision energy
    "bicycle":         2.0,   # ~100kg cyclist → lighter but vulnerable
    "traffic_cone":    0.1,   # static, low mass, ego can displace
    "barrier":         1.0,   # static, but rigid — don't hit it
    "czone_sign":      0.1,
    "generic_object":  0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Intent taxonomy (soft, kinematic-only)
# ─────────────────────────────────────────────────────────────────────────────

class IntentClass(IntEnum):
    UNCOMMITTED = 0
    ASSERTING   = 1
    YIELDING    = 2
    COMMITTED   = 3
    UNAWARE     = 4
    CONTESTED   = 5


def _intent_onehot(intent: IntentClass) -> npt.NDArray[np.float32]:
    v = np.zeros(6, dtype=np.float32)
    v[int(intent)] = 1.0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoadAgentSnapshot:
    """
    Single-frame state of one non-pedestrian road agent, in ego vehicle frame.
    Coordinates are already ego-relative (as stored in annotations.boxes).
    """
    track_token:  str
    agent_type:   str

    # Pose in ego frame
    x:       float       # [m] forward (ego drives in +x direction)
    y:       float       # [m] left
    heading: float       # [rad] relative to ego heading

    # Kinematics
    vx:    float         # [m/s] in ego frame
    vy:    float         # [m/s] in ego frame
    speed: float         # [m/s] scalar

    # Dimensions
    length: float
    width:  float
    height: float

    # Derived scalars (computed once, used many times downstream)
    distance_to_ego:   float   # Euclidean [m]
    approach_speed:    float   # component toward ego (+ve = moving toward ego)
    heading_alignment: float   # cos(agent_heading - ego_heading), 1=same dir, -1=opposing
    half_diagonal:     float   # 0.5 * hypot(length, width) — safety bubble base
    interaction_mass:  float   # type-specific planning weight

    # Type encoding
    is_vehicle:  float
    is_bicycle:  float
    is_static:   float

    # Frame index this was observed (for temporal reasoning)
    frame_idx: int


@dataclass
class RoadAgentTrack:
    """
    Temporal track of one non-pedestrian agent across history frames.
    Snapshots are ordered oldest → newest; gaps allowed (occlusion).
    """
    track_token:  str
    agent_type:   str
    snapshots:    List[RoadAgentSnapshot]

    # Temporal motion descriptors — computed from history
    mean_speed:        float = 0.0
    speed_delta:       float = 0.0   # current_speed - prev_speed  (m/s per 0.5s)
    acceleration_est:  float = 0.0   # speed_delta / HISTORY_DT  [m/s²]
    heading_rate:      float = 0.0   # heading change per second  [rad/s]
    is_stationary:     bool  = False
    is_decelerating:   bool  = False
    is_accelerating:   bool  = False
    is_turning:        bool  = False

    def __post_init__(self) -> None:
        self._compute_motion()

    def _compute_motion(self) -> None:
        if not self.snapshots:
            return

        speeds = [s.speed for s in self.snapshots]
        self.mean_speed    = float(np.mean(speeds))
        self.is_stationary = all(sp < 0.5 for sp in speeds)

        if len(speeds) >= 2:
            self.speed_delta      = speeds[-1] - speeds[-2]
            self.acceleration_est = self.speed_delta / HISTORY_DT
            self.is_decelerating  = self.speed_delta < -0.3   # > 0.6 m/s² decel
            self.is_accelerating  = self.speed_delta >  0.3

        if len(self.snapshots) >= 2:
            dh = self.snapshots[-1].heading - self.snapshots[-2].heading
            # wrap to [-π, π]
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            self.heading_rate = float(dh / HISTORY_DT)
            self.is_turning   = abs(self.heading_rate) > 0.1   # > ~6 deg/s

    @property
    def current(self) -> Optional[RoadAgentSnapshot]:
        """Most recent snapshot (current frame), None if track has no data."""
        return self.snapshots[-1] if self.snapshots else None

    @property
    def num_frames(self) -> int:
        return len(self.snapshots)


@dataclass
class EgoAgentPairFeatures:
    """
    Pairwise interaction features between ego and one road agent.
    This is the core input for intent inference and collision risk.
    """
    track_token:  str
    agent_type:   str

    # Relative position (ego frame, current)
    dx:  float   # [m] agent_x - 0 (ego is at origin)
    dy:  float   # [m] agent_y - 0
    dist: float  # [m] Euclidean

    # Relative velocity (how fast the gap is closing/opening)
    dvx: float   # agent.vx - ego.vx   (+ve = agent moving away in front)
    dvy: float   # agent.vy - ego.vy
    closing_speed: float  # rate at which distance is decreasing (+ve = closing)

    # Euclidean TTC (seconds to closest approach, naive, no lane info)
    ttc_euclidean: float  # capped at 10s if not converging

    # Acceleration-based intent signals (from temporal history)
    agent_accel_est:  float   # [m/s²] from speed_delta / dt
    agent_decel_flag: float   # 1.0 if decelerating toward ego, else 0.0
    agent_accel_flag: float   # 1.0 if accelerating toward ego, else 0.0

    # Heading geometry
    heading_alignment:  float  # cos of relative heading: 1=same, -1=opposing, 0=crossing
    lateral_offset:     float  # |dy| — how far off ego's forward axis the agent is

    # Dimension-based interaction weight
    interaction_mass:  float
    half_diagonal:     float

    # Soft intent one-hot (6-dim) — kinematic fallback without lane data
    intent_onehot: npt.NDArray[np.float32]   # shape (6,)


# ─────────────────────────────────────────────────────────────────────────────
# Core snapshot extraction
# ─────────────────────────────────────────────────────────────────────────────

def _snapshot_from_frame(
    frame: Frame,
    agent_idx: int,
    frame_idx: int,
) -> RoadAgentSnapshot:
    """
    Build one RoadAgentSnapshot from a single frame annotation entry.
    All coordinates remain in ego frame (as stored by NAVSIM).
    """
    ann    = frame.annotations
    box    = ann.boxes[agent_idx]        # [x, y, z, length, width, height, heading]
    vel    = ann.velocity_3d[agent_idx]  # [vx, vy, vz]
    atype  = ann.names[agent_idx]

    x, y       = float(box[0]), float(box[1])
    heading    = float(box[6])
    length     = float(box[3])
    width      = float(box[4])
    height     = float(box[5])
    vx, vy     = float(vel[0]), float(vel[1])
    speed      = float(np.hypot(vx, vy))

    # Approach speed: component of agent velocity pointing toward ego (origin)
    dist = float(np.hypot(x, y))
    if dist > 1e-3:
        # unit vector from agent toward ego = (-x/dist, -y/dist)
        approach_speed = float(-(x * vx + y * vy) / dist)
    else:
        approach_speed = 0.0

    # Heading alignment with ego (ego heading = 0 in ego frame)
    heading_alignment = float(np.cos(heading))   # 1 = same dir, -1 = opposing

    return RoadAgentSnapshot(
        track_token       = ann.track_tokens[agent_idx],
        agent_type        = atype,
        x                 = x,
        y                 = y,
        heading           = heading,
        vx                = vx,
        vy                = vy,
        speed             = speed,
        length            = length,
        width             = width,
        height            = height,
        distance_to_ego   = dist,
        approach_speed    = approach_speed,
        heading_alignment = heading_alignment,
        half_diagonal     = 0.5 * float(np.hypot(length, width)),
        interaction_mass  = INTERACTION_MASS.get(atype, 0.5),
        is_vehicle        = float(atype == TYPE_VEHICLE),
        is_bicycle        = float(atype == TYPE_BICYCLE),
        is_static         = float(atype in TYPE_STATIC),
        frame_idx         = frame_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal track builder
# ─────────────────────────────────────────────────────────────────────────────

def extract_road_agent_tracks(
    agent_input: AgentInput,
    agent_types: Optional[frozenset] = None,
    max_distance: float = 80.0,
) -> Dict[str, List[RoadAgentTrack]]:
    """
    Extract temporal tracks for non-pedestrian road agents, grouped by type.

    Args:
        agent_input:  NAVSIM AgentInput
        agent_types:  Which types to extract. Defaults to all non-pedestrian types.
        max_distance: Only include agents within this radius of ego [m].
                      Vehicles typically need 80m for highway/intersection scenarios.

    Returns:
        Dict[agent_type → List[RoadAgentTrack]], sorted by distance per group.

    Key design choices vs naive extraction:
      - Static objects (cones, barriers) have zero velocity — their track history
        is still useful because it confirms they ARE static (no delta across frames).
        A cone that "moves" between frames is a data artifact or occlusion; the history
        exposes this.
      - Bicycles are tracked separately from vehicles because their kinematic envelope
        is much smaller (can stop faster, can be on sidewalk) and they're more
        vulnerable than vehicles but less so than pedestrians.
      - track_token matching is critical for both dynamic and static types.
        A barrier that disappears between frames (occlusion) should be preserved as
        a gap in the track, not discarded — the model needs to know it existed.
    """
    if agent_types is None:
        agent_types = ALL_NON_PED

    frames = agent_input.frames

    # Collect per-token snapshots across all frames
    token_snapshots: Dict[str, Dict[int, RoadAgentSnapshot]] = defaultdict(dict)
    token_types:     Dict[str, str] = {}

    for frame_idx, frame in enumerate(frames):
        ann = frame.annotations
        for i, name in enumerate(ann.names):
            if name not in agent_types:
                continue
            snapshot = _snapshot_from_frame(frame, i, frame_idx)
            if snapshot.distance_to_ego > max_distance:
                continue
            token = snapshot.track_token
            token_snapshots[token][frame_idx] = snapshot
            token_types[token] = name

    # Assemble RoadAgentTrack per token
    result: Dict[str, List[RoadAgentTrack]] = {t: [] for t in agent_types}

    for token, frame_snap_map in token_snapshots.items():
        ordered = [frame_snap_map[fi] for fi in sorted(frame_snap_map.keys())]
        atype   = token_types[token]
        track   = RoadAgentTrack(
            track_token = token,
            agent_type  = atype,
            snapshots   = ordered,
        )
        result[atype].append(track)

    # Sort each group by current-frame distance (closest first)
    def _dist_key(t: RoadAgentTrack) -> float:
        return t.current.distance_to_ego if t.current else 999.0

    for atype in result:
        result[atype].sort(key=_dist_key)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise interaction features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_euclidean_ttc(
    dx: float, dy: float,
    dvx: float, dvy: float,
    ttc_cap: float = 10.0,
) -> float:
    """
    Time to closest approach (Euclidean, no lane geometry).
    This is the fallback TTC described in the design doc when lane data
    is not available. It underestimates risk for crossing scenarios
    but is cheap and direction-independent.

    Formula: TTC = -( r · v_rel ) / |v_rel|²  where r = (dx, dy)
    Positive TTC means they are converging; negative means diverging.
    """
    v_sq = dvx * dvx + dvy * dvy
    if v_sq < 1e-6:
        return ttc_cap   # not converging

    dot_rv = dx * dvx + dy * dvy
    if dot_rv >= 0:
        return ttc_cap   # relative motion is away from each other

    ttc = -dot_rv / v_sq
    return float(min(ttc, ttc_cap))


def _kinematic_intent(
    track: RoadAgentTrack,
    snapshot: RoadAgentSnapshot,
    ttc: float,
) -> IntentClass:
    """
    Soft intent classification from kinematics alone.
    This is the fallback mode (no lane data), as described in the design doc.

    Rules (in priority order):
      COMMITTED   — already very close (dist < 2× half_diagonal) AND moving
      YIELDING    — decelerating AND approach_speed > 0 (closing toward ego)
      ASSERTING   — accelerating AND approach_speed > 0 AND TTC < 5s
      UNAWARE     — no speed change despite being on converging path (TTC < 5s)
      UNCOMMITTED — TTC > 5s, no clear signal
      CONTESTED   — both asserting: agent accel + TTC < 2s (danger zone)

    Limitations without lane data:
      - Cannot distinguish "asserting through green light" from "running red"
      - Cannot detect contested states from other-agent pairs (only ego-agent)
      - UNAWARE vs ASSERTING is ambiguous for constant-speed agents;
        we label as ASSERTING if approach is fast, UNAWARE if slow
    """
    dist = snapshot.distance_to_ego
    committed_threshold = 2.0 * snapshot.half_diagonal + 0.5 * (EGO_LENGTH + EGO_WIDTH)

    # CONTESTED: very close + agent accelerating
    if dist < committed_threshold and track.is_accelerating:
        return IntentClass.CONTESTED

    # COMMITTED: inside the conflict zone (already too close to matter)
    if dist < committed_threshold:
        return IntentClass.COMMITTED

    converging = snapshot.approach_speed > 0.5 and ttc < 8.0

    # YIELDING: decelerating while on converging path
    if track.is_decelerating and converging:
        return IntentClass.YIELDING

    # ASSERTING: accelerating + converging + not far away
    if track.is_accelerating and converging and ttc < 5.0:
        return IntentClass.ASSERTING

    # UNAWARE: converging but no speed change at all
    if converging and not track.is_accelerating and not track.is_decelerating and ttc < 5.0:
        # Distinguish from ASSERTING: approach_speed must be substantial
        if snapshot.approach_speed > 2.0:
            return IntentClass.UNAWARE
        return IntentClass.ASSERTING   # mild constant-speed asserting

    # UNCOMMITTED: default
    return IntentClass.UNCOMMITTED


def compute_ego_agent_pair_features(
    track: RoadAgentTrack,
    ego_vx: float,
    ego_vy: float,
) -> Optional[EgoAgentPairFeatures]:
    """
    Compute pairwise interaction features between ego and one road agent.

    Args:
        track:   The agent's temporal track
        ego_vx:  Ego forward velocity [m/s] in ego frame
        ego_vy:  Ego lateral velocity [m/s] in ego frame

    Returns:
        EgoAgentPairFeatures, or None if track has no current observation.

    Note on ego velocity in ego frame:
        In ego frame, ego is always at origin with heading=0.
        ego_vx ≈ ego_speed (forward component), ego_vy ≈ 0 normally.
        We use the full vector for correctness during turns.
    """
    snap = track.current
    if snap is None:
        return None

    dx, dy = snap.x, snap.y
    dist   = snap.distance_to_ego

    # Relative velocity: agent velocity minus ego velocity
    dvx = snap.vx - ego_vx
    dvy = snap.vy - ego_vy

    # Closing speed: rate at which the inter-agent distance is decreasing
    # Positive = closing, Negative = opening
    if dist > 1e-3:
        closing_speed = float(-(dx * dvx + dy * dvy) / dist)
    else:
        closing_speed = 0.0

    ttc = _compute_euclidean_ttc(dx, dy, dvx, dvy)

    # Intent from kinematics
    intent = _kinematic_intent(track, snap, ttc)

    return EgoAgentPairFeatures(
        track_token        = snap.track_token,
        agent_type         = snap.agent_type,
        dx                 = dx,
        dy                 = dy,
        dist               = dist,
        dvx                = dvx,
        dvy                = dvy,
        closing_speed      = closing_speed,
        ttc_euclidean      = ttc,
        agent_accel_est    = track.acceleration_est,
        agent_decel_flag   = float(track.is_decelerating),
        agent_accel_flag   = float(track.is_accelerating),
        heading_alignment  = snap.heading_alignment,
        lateral_offset     = abs(dy),
        interaction_mass   = snap.interaction_mass,
        half_diagonal      = snap.half_diagonal,
        intent_onehot      = _intent_onehot(intent),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversion
# ─────────────────────────────────────────────────────────────────────────────

def tracks_to_tensor(
    tracks: List[RoadAgentTrack],
    num_agents: int  = 32,
    num_frames: int  = 4,
    device: str      = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert RoadAgentTrack list to padded tensor for model input.

    Output shape: [num_agents, num_frames, FEATURE_DIM]

    Feature vector per (agent, frame) — 20 dims:
        [ 0]  x                    ego-frame forward position [m]
        [ 1]  y                    ego-frame lateral position [m]
        [ 2]  sin(heading)         avoids ±π discontinuity
        [ 3]  cos(heading)
        [ 4]  vx                   forward velocity [m/s]
        [ 5]  vy                   lateral velocity [m/s]
        [ 6]  speed                scalar [m/s]
        [ 7]  length               box length [m]
        [ 8]  width                box width [m]
        [ 9]  height               box height [m]
        [10]  distance_to_ego      Euclidean [m]
        [11]  approach_speed       component toward ego [m/s]
        [12]  heading_alignment    cos(agent_hdg - ego_hdg)
        [13]  half_diagonal        safety bubble base [m]
        [14]  interaction_mass     type-specific planning weight
        [15]  is_vehicle           type one-hot
        [16]  is_bicycle           type one-hot
        [17]  is_static            type one-hot
        [18]  was_seen_prev_frame  1.0 if also visible in previous frame
        [19]  frame_age            how many frames ago this was (0=current, 1=prev, ...)
        ─────────────────────────────────────────────
        Total: 20 features

    Also returns:
        valid_mask: [num_agents, num_frames] bool — True where real data exists

    Two extra features not in the pedestrian extractor:
      [18] was_seen_prev_frame — critical for static-object disambiguation:
           a barrier visible in all 4 frames is confirmed static; one that
           appears only in frame 3 might be newly placed or a false detection.
      [19] frame_age — lets the model weight recent frames more in temporal attention.
    """
    FEATURE_DIM = 20
    features = np.zeros((num_agents, num_frames, FEATURE_DIM), dtype=np.float32)
    valid    = np.zeros((num_agents, num_frames), dtype=bool)

    for agent_idx, track in enumerate(tracks[:num_agents]):
        n_snaps    = len(track.snapshots)
        start_slot = num_frames - n_snaps

        snap_frame_indices = {s.frame_idx for s in track.snapshots}

        for snap_offset, snapshot in enumerate(track.snapshots):
            frame_slot = start_slot + snap_offset
            if frame_slot < 0:
                continue
            if frame_slot >= num_frames:
                break

            # Was the agent also seen in the previous frame?
            was_seen_prev = float(
                (snapshot.frame_idx - 1) in snap_frame_indices
            )
            frame_age = float(num_frames - 1 - frame_slot)   # 0 = current frame

            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            f[ 0] = snapshot.x
            f[ 1] = snapshot.y
            f[ 2] = np.sin(snapshot.heading)
            f[ 3] = np.cos(snapshot.heading)
            f[ 4] = snapshot.vx
            f[ 5] = snapshot.vy
            f[ 6] = snapshot.speed
            f[ 7] = snapshot.length
            f[ 8] = snapshot.width
            f[ 9] = snapshot.height
            f[10] = snapshot.distance_to_ego
            f[11] = snapshot.approach_speed
            f[12] = snapshot.heading_alignment
            f[13] = snapshot.half_diagonal
            f[14] = snapshot.interaction_mass
            f[15] = snapshot.is_vehicle
            f[16] = snapshot.is_bicycle
            f[17] = snapshot.is_static
            f[18] = was_seen_prev
            f[19] = frame_age

            features[agent_idx, frame_slot] = f
            valid[agent_idx, frame_slot]    = True

    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(valid).to(device),
    )


def pairwise_to_tensor(
    pair_features: List[EgoAgentPairFeatures],
    num_agents: int = 32,
    device: str     = "cpu",
) -> torch.Tensor:
    """
    Convert list of EgoAgentPairFeatures to a padded tensor.

    Output shape: [num_agents, PAIR_DIM]

    Feature vector per agent — 18 dims:
        [ 0]  dx                   relative x [m]
        [ 1]  dy                   relative y [m]
        [ 2]  dist                 Euclidean distance [m]
        [ 3]  dvx                  relative vx (agent - ego) [m/s]
        [ 4]  dvy                  relative vy (agent - ego) [m/s]
        [ 5]  closing_speed        rate of gap closure [m/s]
        [ 6]  ttc_euclidean        time to closest approach [s] (capped at 10s)
        [ 7]  agent_accel_est      estimated acceleration [m/s²]
        [ 8]  agent_decel_flag     1 if decelerating
        [ 9]  agent_accel_flag     1 if accelerating
        [10]  heading_alignment    cos(relative heading)
        [11]  lateral_offset       |dy| [m]
        [12]  interaction_mass     type-based planning weight
        [13]  half_diagonal        safety bubble [m]
        [14-19]  intent_onehot     6-class soft intent
        ─────────────────────────────────────────────
        Total: 20 features (14 scalars + 6 intent)
    """
    PAIR_DIM = 20
    out = np.zeros((num_agents, PAIR_DIM), dtype=np.float32)

    for i, pf in enumerate(pair_features[:num_agents]):
        out[i,  0] = pf.dx
        out[i,  1] = pf.dy
        out[i,  2] = pf.dist
        out[i,  3] = pf.dvx
        out[i,  4] = pf.dvy
        out[i,  5] = pf.closing_speed
        out[i,  6] = pf.ttc_euclidean
        out[i,  7] = pf.agent_accel_est
        out[i,  8] = pf.agent_decel_flag
        out[i,  9] = pf.agent_accel_flag
        out[i, 10] = pf.heading_alignment
        out[i, 11] = pf.lateral_offset
        out[i, 12] = pf.interaction_mass
        out[i, 13] = pf.half_diagonal
        out[i, 14:20] = pf.intent_onehot

    return torch.from_numpy(out).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Builder (plug-in for NAVSIM AbstractFeatureBuilder)
# ─────────────────────────────────────────────────────────────────────────────

class InteractionFeatureBuilder:
    """
    NAVSIM-compatible feature builder for all non-pedestrian road agents.

    Produces five tensors:
      "vehicle_features"   [N_v, T, 20]   vehicle temporal states
      "vehicle_valid"      [N_v, T]        validity mask
      "bicycle_features"   [N_b, T, 20]   bicycle temporal states
      "bicycle_valid"      [N_b, T]        validity mask
      "static_features"    [N_s, T, 20]   static object temporal states
      "static_valid"       [N_s, T]        validity mask
      "vehicle_pairs"      [N_v, 20]       ego-vehicle pairwise features + intent
      "bicycle_pairs"      [N_b, 20]       ego-bicycle pairwise features + intent

    Note: static objects don't get pairwise intent features (they have no intent).
    They are included in temporal tracks to confirm they ARE static across frames.
    """

    def __init__(
        self,
        max_vehicles:   int   = 32,
        max_bicycles:   int   = 8,
        max_static:     int   = 16,
        num_history:    int   = 4,
        max_distance:   float = 80.0,
    ):
        self.max_vehicles  = max_vehicles
        self.max_bicycles  = max_bicycles
        self.max_static    = max_static
        self.num_history   = num_history
        self.max_distance  = max_distance

    def get_unique_name(self) -> str:
        return "interaction_features"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_status = agent_input.ego_statuses[-1]
        ego_vx = float(ego_status.ego_velocity[0])
        ego_vy = float(ego_status.ego_velocity[1])

        # ── 1. Extract temporal tracks ──────────────────────────────────────
        tracks = extract_road_agent_tracks(
            agent_input,
            max_distance=self.max_distance,
        )

        vehicle_tracks = tracks.get(TYPE_VEHICLE, [])
        bicycle_tracks = tracks.get(TYPE_BICYCLE, [])
        # All static subtypes merged into one list, sorted by distance
        static_tracks  = sorted(
            [t for atype in TYPE_STATIC for t in tracks.get(atype, [])],
            key=lambda t: t.current.distance_to_ego if t.current else 999.0,
        )

        # ── 2. Temporal tensors ──────────────────────────────────────────────
        v_feat, v_valid = tracks_to_tensor(
            vehicle_tracks, num_agents=self.max_vehicles, num_frames=self.num_history
        )
        b_feat, b_valid = tracks_to_tensor(
            bicycle_tracks, num_agents=self.max_bicycles, num_frames=self.num_history
        )
        s_feat, s_valid = tracks_to_tensor(
            static_tracks, num_agents=self.max_static, num_frames=self.num_history
        )

        # ── 3. Pairwise interaction features (dynamic agents only) ───────────
        v_pairs_raw = [
            pf for t in vehicle_tracks[:self.max_vehicles]
            if (pf := compute_ego_agent_pair_features(t, ego_vx, ego_vy)) is not None
        ]
        b_pairs_raw = [
            pf for t in bicycle_tracks[:self.max_bicycles]
            if (pf := compute_ego_agent_pair_features(t, ego_vx, ego_vy)) is not None
        ]

        v_pairs = pairwise_to_tensor(v_pairs_raw, num_agents=self.max_vehicles)
        b_pairs = pairwise_to_tensor(b_pairs_raw, num_agents=self.max_bicycles)

        return {
            "vehicle_features":  v_feat,
            "vehicle_valid":     v_valid,
            "bicycle_features":  b_feat,
            "bicycle_valid":     b_valid,
            "static_features":   s_feat,
            "static_valid":      s_valid,
            "vehicle_pairs":     v_pairs,
            "bicycle_pairs":     b_pairs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utility: merge all agent types into one flat tensor (for simpler models)
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_agents_tensor(
    agent_input: AgentInput,
    num_agents:  int   = 64,
    num_frames:  int   = 4,
    max_distance: float = 80.0,
    device: str        = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single flat tensor of ALL non-pedestrian agents, sorted by distance.
    Useful for simpler attention models that don't need type-separated inputs.

    Output:
        features: [num_agents, num_frames, 20]
        valid:    [num_agents, num_frames]
    """
    tracks = extract_road_agent_tracks(agent_input, max_distance=max_distance)
    all_tracks = []
    for atype_tracks in tracks.values():
        all_tracks.extend(atype_tracks)

    # Sort all agents together by distance
    all_tracks.sort(
        key=lambda t: t.current.distance_to_ego if t.current else 999.0
    )

    return tracks_to_tensor(all_tracks, num_agents=num_agents, num_frames=num_frames, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Quick reference printout
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("InteractionFeatureExtractor — layout reference")
    print("=" * 65)

    print("\nTemporal feature vector [per agent, per frame] — 20 dims:")
    temporal_feats = [
        "x",               "y",               "sin(heading)",      "cos(heading)",
        "vx",              "vy",              "speed",
        "length",          "width",           "height",
        "distance_to_ego", "approach_speed",  "heading_alignment",
        "half_diagonal",   "interaction_mass",
        "is_vehicle",      "is_bicycle",      "is_static",
        "was_seen_prev_frame", "frame_age",
    ]
    for i, name in enumerate(temporal_feats):
        print(f"  [{i:2d}] {name}")

    print("\nPairwise (ego-agent) feature vector — 20 dims:")
    pair_feats = [
        "dx",             "dy",             "dist",
        "dvx",            "dvy",            "closing_speed",
        "ttc_euclidean",  "agent_accel_est",
        "agent_decel_flag", "agent_accel_flag",
        "heading_alignment", "lateral_offset",
        "interaction_mass",  "half_diagonal",
        "intent[UNCOMMITTED]", "intent[ASSERTING]", "intent[YIELDING]",
        "intent[COMMITTED]", "intent[UNAWARE]", "intent[CONTESTED]",
    ]
    for i, name in enumerate(pair_feats):
        print(f"  [{i:2d}] {name}")

    print("\nInteraction mass by type (planning weight):")
    for atype, mass in INTERACTION_MASS.items():
        l, w, _ = DIMENSION_PRIORS[atype]
        hd = 0.5 * np.hypot(l, w)
        print(f"  {atype:<16s}  mass={mass:4.1f}  half_diagonal={hd:.2f}m")

    print("\nIntent taxonomy (kinematic fallback, no lane data):")
    for ic in IntentClass:
        print(f"  [{ic.value}] {ic.name}")