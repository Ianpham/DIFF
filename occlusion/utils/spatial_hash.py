"""Spatial hash grid for efficient Gaussian neighbor queries.

Partitions the 3D scene into coarse cells and assigns each Gaussian
to a cell. For any query point, only Gaussians in the same and
adjacent cells (3x3x3 = 27 neighborhood) need to be checked.

With 25,600 Gaussians, cell_size=4m, and scene 80x80x6.4m:
    Grid: 20x20x2 = 800 cells
    Avg Gaussians per cell: ~32
    Neighbors to check: ~32 * 27 = 864  (vs 25,600 brute force)
    Speedup: ~30x

All operations are pure PyTorch on GPU. No external dependencies.
"""

import torch
from typing import Tuple, Optional, List


class SpatialHashGrid:
    """GPU-friendly spatial hash for batched 3D point sets.

    Not an nn.Module — this is a stateless utility rebuilt each forward
    pass from the current Gaussian means. No learned parameters.

    Usage:
        grid = SpatialHashGrid(means, cell_size, pc_range)
        neighbors, neighbor_mask = grid.query(voxel_positions, radius)
    """

    def __init__(
        self,
        points: torch.Tensor,
        cell_size: float,
        point_cloud_range: List[float],
        max_points_per_cell: int = 128,
    ):
        """Build spatial hash from a set of 3D points.

        Args:
            points: (B, N, 3) point positions (e.g. Gaussian means).
            cell_size: size of each hash cell in meters.
                Should be >= neighbor_radius for correct 27-cell lookup.
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
            max_points_per_cell: cap per cell to keep memory bounded.
                Points beyond this are dropped (rare with good cell_size).
        """
        self.device = points.device
        self.B = points.shape[0]
        self.N = points.shape[1]
        self.cell_size = cell_size
        self.max_per_cell = max_points_per_cell

        pc = torch.tensor(point_cloud_range, device=self.device, dtype=torch.float32)
        self.origin = pc[:3]  # (3,)
        self.range_max = pc[3:]  # (3,)

        # grid dimensions
        extent = self.range_max - self.origin
        self.grid_dims = torch.ceil(extent / cell_size).long().clamp(min=1)
        # grid_dims: (3,) — [Gx, Gy, Gz]

        self.Gx = self.grid_dims[0].item()
        self.Gy = self.grid_dims[1].item()
        self.Gz = self.grid_dims[2].item()
        self.num_cells = self.Gx * self.Gy * self.Gz

        # build the index
        self._build(points)

    def _pos_to_cell(self, pos: torch.Tensor) -> torch.Tensor:
        """Convert 3D positions to cell indices.

        Args:
            pos: (..., 3) positions.

        Returns:
            cell_idx: (..., 3) integer cell coordinates, clamped to grid.
        """
        cell = ((pos - self.origin) / self.cell_size).long()
        cell[..., 0] = cell[..., 0].clamp(0, self.Gx - 1)
        cell[..., 1] = cell[..., 1].clamp(0, self.Gy - 1)
        cell[..., 2] = cell[..., 2].clamp(0, self.Gz - 1)
        return cell

    def _cell_to_flat(self, cell: torch.Tensor) -> torch.Tensor:
        """Convert 3D cell coordinates to flat index.

        Args:
            cell: (..., 3) integer cell coords.

        Returns:
            flat: (...) flat index.
        """
        return (
            cell[..., 0] * (self.Gy * self.Gz)
            + cell[..., 1] * self.Gz
            + cell[..., 2]
        )

    def _build(self, points: torch.Tensor):
        """Assign points to cells and build lookup tables.

        Creates:
            cell_contents: (B, num_cells, max_per_cell) point indices per cell.
            cell_counts: (B, num_cells) number of points in each cell.
        """
        B, N, _ = points.shape
        device = self.device
        M = self.max_per_cell

        cell_coords = self._pos_to_cell(points)  # (B, N, 3)
        flat_idx = self._cell_to_flat(cell_coords)  # (B, N)

        # initialize storage
        self.cell_contents = torch.full(
            (B, self.num_cells, M), -1, dtype=torch.long, device=device
        )
        self.cell_counts = torch.zeros(
            B, self.num_cells, dtype=torch.long, device=device
        )

        # assign points to cells
        # scatter approach: for each point, atomically add to its cell
        for b in range(B):
            cells_b = flat_idx[b]  # (N,)

            # sort by cell for efficient sequential assignment
            sorted_order = cells_b.argsort()
            sorted_cells = cells_b[sorted_order]

            # find boundaries
            changes = torch.cat([
                torch.tensor([0], device=device),
                (sorted_cells[1:] != sorted_cells[:-1]).nonzero(as_tuple=False).squeeze(-1) + 1,
                torch.tensor([N], device=device),
            ])

            for i in range(changes.shape[0] - 1):
                start = changes[i].item()
                end = changes[i + 1].item()
                cell_id = sorted_cells[start].item()
                count = min(end - start, M)
                self.cell_contents[b, cell_id, :count] = sorted_order[start:start + count]
                self.cell_counts[b, cell_id] = count

    def query_neighbors(
        self,
        query_points: torch.Tensor,
        radius_sq: float,
        source_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find source points within radius of each query point.

        Uses 27-cell neighborhood lookup.

        Args:
            query_points: (V, 3) voxel positions (no batch dim — same for all).
            radius_sq: squared neighbor radius.
            source_points: (B, N, 3) the original points used in __init__.

        Returns:
            neighbor_indices: (B, V, max_neighbors) indices into source_points.
                -1 for padding.
            neighbor_counts: (B, V) number of actual neighbors per query.
        """
        V = query_points.shape[0]
        B = self.B
        device = self.device

        # determine max possible neighbors: 27 cells * max_per_cell
        max_neighbors = min(27 * self.max_per_cell, self.N)

        neighbor_indices = torch.full(
            (B, V, max_neighbors), -1, dtype=torch.long, device=device
        )
        neighbor_counts = torch.zeros(B, V, dtype=torch.long, device=device)

        # cell coords for query points
        query_cells = self._pos_to_cell(query_points)  # (V, 3)

        # 27-neighborhood offsets
        offsets = torch.tensor(
            [[dx, dy, dz]
             for dx in [-1, 0, 1]
             for dy in [-1, 0, 1]
             for dz in [-1, 0, 1]],
            device=device, dtype=torch.long,
        )  # (27, 3)

        # for each query, gather candidate cells
        # query_cells: (V, 3) + offsets: (27, 3) -> neighbor_cells: (V, 27, 3)
        neighbor_cells = query_cells.unsqueeze(1) + offsets.unsqueeze(0)

        # clamp to grid bounds
        neighbor_cells[..., 0] = neighbor_cells[..., 0].clamp(0, self.Gx - 1)
        neighbor_cells[..., 1] = neighbor_cells[..., 1].clamp(0, self.Gy - 1)
        neighbor_cells[..., 2] = neighbor_cells[..., 2].clamp(0, self.Gz - 1)

        # flat cell indices: (V, 27)
        flat_neighbor_cells = self._cell_to_flat(neighbor_cells)

        # gather candidates and distance-check
        # process per batch item since cell_contents is per-batch
        for b in range(B):
            for v_start in range(0, V, 1000):
                v_end = min(v_start + 1000, V)
                v_count = v_end - v_start

                # candidate cell contents: (v_count, 27, max_per_cell)
                cell_flat = flat_neighbor_cells[v_start:v_end]  # (v_count, 27)
                candidates = self.cell_contents[b][cell_flat]  # (v_count, 27, M)
                cand_counts = self.cell_counts[b][cell_flat]  # (v_count, 27)

                # flatten: (v_count, 27*M)
                candidates_flat = candidates.reshape(v_count, -1)
                valid_cand = candidates_flat >= 0  # (v_count, 27*M)

                # distance check
                qp = query_points[v_start:v_end]  # (v_count, 3)

                # gather candidate positions
                # replace -1 with 0 for safe gather, mask later
                safe_idx = candidates_flat.clamp(min=0)  # (v_count, 27*M)
                cand_pos = source_points[b][safe_idx]  # (v_count, 27*M, 3)

                diff = qp.unsqueeze(1) - cand_pos  # (v_count, 27*M, 3)
                dist_sq = (diff ** 2).sum(dim=-1)  # (v_count, 27*M)

                # valid = in range AND not padding
                in_range = (dist_sq < radius_sq) & valid_cand

                # collect valid neighbors per query
                for vi in range(v_count):
                    valid_mask = in_range[vi]  # (27*M,)
                    valid_idx = candidates_flat[vi][valid_mask]
                    n = min(valid_idx.shape[0], max_neighbors)
                    neighbor_indices[b, v_start + vi, :n] = valid_idx[:n]
                    neighbor_counts[b, v_start + vi] = n

        return neighbor_indices, neighbor_counts


    def query_neighbors_vectorized(
        self,
        query_points: torch.Tensor,
        radius_sq: float,
        source_points: torch.Tensor,
        max_neighbors: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized neighbor query — no Python loops over voxels.

        Trades memory for speed. Instead of exact per-voxel neighbor lists,
        returns a fixed-size candidate set per voxel with a validity mask.

        Args:
            query_points: (V, 3).
            radius_sq: squared search radius.
            source_points: (B, N, 3).
            max_neighbors: fixed output size per query.

        Returns:
            candidate_indices: (B, V, max_neighbors) point indices. Padded with 0.
            candidate_mask: (B, V, max_neighbors) bool. True = valid neighbor.
            candidate_dist_sq: (B, V, max_neighbors) squared distances.
        """
        V = query_points.shape[0]
        B = self.B
        device = self.device
        M = self.max_per_cell

        # 27-neighborhood cell lookup: (V, 27, 3) -> (V, 27)
        query_cells = self._pos_to_cell(query_points)
        offsets = torch.tensor(
            [[dx, dy, dz]
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]],
            device=device, dtype=torch.long,
        )
        neighbor_cells = query_cells.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_cells[..., 0] = neighbor_cells[..., 0].clamp(0, self.Gx - 1)
        neighbor_cells[..., 1] = neighbor_cells[..., 1].clamp(0, self.Gy - 1)
        neighbor_cells[..., 2] = neighbor_cells[..., 2].clamp(0, self.Gz - 1)
        flat_cells = self._cell_to_flat(neighbor_cells)  # (V, 27)

        # total candidates per voxel: 27 * M
        total_cand = 27 * M

        all_candidate_indices = []
        all_candidate_mask = []
        all_candidate_dist_sq = []

        for b in range(B):
            # gather cell contents for all voxels at once
            # cell_contents[b]: (num_cells, M)
            # flat_cells: (V, 27)
            # gathered: (V, 27, M)
            gathered = self.cell_contents[b][flat_cells]
            gathered_flat = gathered.reshape(V, total_cand)  # (V, 27*M)

            valid_cand = gathered_flat >= 0  # (V, 27*M)
            safe_idx = gathered_flat.clamp(min=0)  # (V, 27*M)

            # gather positions: (V, 27*M, 3)
            cand_pos = source_points[b][safe_idx]

            # distances: (V, 27*M)
            diff = query_points.unsqueeze(1) - cand_pos
            dist_sq = (diff ** 2).sum(dim=-1)

            # valid = exists AND in range
            in_range = (dist_sq < radius_sq) & valid_cand  # (V, 27*M)

            # take top-k closest valid neighbors
            # set invalid distances to inf
            dist_sq_masked = dist_sq.clone()
            dist_sq_masked[~in_range] = float("inf")

            k = min(max_neighbors, total_cand)
            topk_dist, topk_idx_in_cand = torch.topk(
                dist_sq_masked, k, dim=-1, largest=False
            )  # (V, k)

            # map back to point indices
            topk_point_idx = torch.gather(safe_idx, 1, topk_idx_in_cand)  # (V, k)
            topk_valid = topk_dist < float("inf")  # (V, k)

            all_candidate_indices.append(topk_point_idx)
            all_candidate_mask.append(topk_valid)
            all_candidate_dist_sq.append(topk_dist)

        candidate_indices = torch.stack(all_candidate_indices, dim=0)  # (B, V, k)
        candidate_mask = torch.stack(all_candidate_mask, dim=0)  # (B, V, k)
        candidate_dist_sq = torch.stack(all_candidate_dist_sq, dim=0)  # (B, V, k)

        return candidate_indices, candidate_mask, candidate_dist_sq
    
def build_spatial_hash(
    gaussian_means: torch.Tensor,
    neighbor_radius: float,
    point_cloud_range: List[float],
    max_points_per_cell: int = 128,
) -> SpatialHashGrid:
    """Convenience constructor.

    Args:
        gaussian_means: (B, N, 3).
        neighbor_radius: search radius in meters.
        point_cloud_range: scene bounds.
        max_points_per_cell: memory cap.

    Returns:
        SpatialHashGrid instance.
    """
    # cell size >= radius so 27-cell neighborhood covers the search sphere
    cell_size = max(neighbor_radius, 1.0)
    return SpatialHashGrid(
        gaussian_means, cell_size, point_cloud_range, max_points_per_cell,
    )
