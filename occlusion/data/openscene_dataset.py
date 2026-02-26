"""OpenScene dataset loader with occupancy labels for NavSim v1."""
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset
from typing import Dict, Tuple

class OpenSceneOccDataset(Dataset):
    CAMERA_NAMES = ["CAM_F0","CAM_L0","CAM_R0","CAM_L1","CAM_R1","CAM_L2","CAM_R2","CAM_B0"]

    def __init__(self, data_root, occ_label_root, split="train", info_file=None,
                 img_size=(448,800), point_cloud_range=None, occ_size=None,
                 num_classes=17, trajectory_length=8, history_length=4,
                 load_occ=True, load_planning=True):
        self.data_root = data_root; self.occ_root = occ_label_root; self.split = split
        self.img_size = img_size; self.num_classes = num_classes
        self.traj_len = trajectory_length; self.hist_len = history_length
        self.load_occ = load_occ; self.load_planning = load_planning
        self.pc_range = np.array(point_cloud_range or [-40,-40,-1,40,40,5.4])
        self.occ_size = occ_size or [200,200,16]
        if info_file is None: info_file = os.path.join(data_root, f"openscene_infos_{split}.pkl")
        if os.path.exists(info_file):
            with open(info_file, "rb") as f: self.infos = pickle.load(f)
        else:
            print(f"Warning: {info_file} not found. Using {100 if split=='train' else 20} dummy samples.")
            self.infos = self._dummy(100 if split=="train" else 20)

    def _dummy(self, n):
        infos = []
        for i in range(n):
            info = {"token": f"s_{i:06d}", "lidar_path": f"dummy_{i}.bin",
                    "ego_status": dict(speed=np.random.uniform(0,15), acceleration=np.random.uniform(-3,3),
                                       yaw_rate=np.random.uniform(-0.3,0.3), steering_angle=0.0),
                    "cams": {c: {"data_path": f"d_{c}_{i}.jpg",
                                 "cam_intrinsic": np.eye(3,dtype=np.float32)*1000,
                                 "lidar2img": np.eye(4,dtype=np.float32)} for c in self.CAMERA_NAMES}}
            if self.load_planning:
                info["gt_trajectory"] = np.random.randn(self.traj_len,2).astype(np.float32)*2
                info["history_trajectory"] = np.random.randn(self.hist_len,2).astype(np.float32)
            infos.append(info)
        return infos

    def __len__(self): return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]; data = {}
        imgs, l2i = [], []
        for cn in self.CAMERA_NAMES:
            ci = info["cams"][cn]
            ip = os.path.join(self.data_root, "sensor_blobs", ci["data_path"])
            img = np.random.randint(0,256,(*self.img_size,3),dtype=np.uint8).astype(np.float32) if not os.path.exists(ip) else np.array(__import__("PIL.Image", fromlist=["Image"]).open(ip).resize((self.img_size[1],self.img_size[0]))).astype(np.float32)
            img -= np.array([103.530,116.280,123.675]); imgs.append(img); l2i.append(ci["lidar2img"])
        data["images"] = torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float()
        data["lidar2img"] = torch.from_numpy(np.stack(l2i)).float()
        lp = os.path.join(self.data_root, "sensor_blobs", info["lidar_path"])
        if os.path.exists(lp): pts = np.fromfile(lp, dtype=np.float32).reshape(-1,5)
        else:
            n = np.random.randint(30000,60000)
            pts = np.concatenate([np.random.uniform(self.pc_range[:3],self.pc_range[3:],(n,3)),
                                  np.random.uniform(0,1,(n,1)), np.zeros((n,1))], axis=1).astype(np.float32)
        data["points"] = torch.from_numpy(pts).float()
        if self.load_occ:
            op = os.path.join(self.occ_root, f"{info['token']}.npz")
            if os.path.exists(op): od = np.load(op); ol = od["occ_label"]
            else: ol = np.zeros(self.occ_size, dtype=np.int64)
            data["occ_label"] = torch.from_numpy(ol).long()
        e = info["ego_status"]
        data["ego_status"] = torch.tensor([e.get("speed",0),e.get("acceleration",0),e.get("yaw_rate",0),e.get("steering_angle",0),0,0,0], dtype=torch.float32)
        if self.load_planning:
            gt = info.get("gt_trajectory", np.zeros((self.traj_len,2),dtype=np.float32))
            ht = info.get("history_trajectory", np.zeros((self.hist_len,2),dtype=np.float32))
            data["gt_trajectory"] = torch.from_numpy(gt).float()
            data["gt_actions"] = torch.from_numpy(np.diff(gt, axis=0, prepend=gt[:1])).float()
            data["history_trajectory"] = torch.from_numpy(ht).float()
        data["token"] = info["token"]; data["img_shape"] = torch.tensor(self.img_size)
        return data

def collate_fn(batch):
    c = {}
    for k in ["images","lidar2img","occ_label","ego_status","gt_trajectory","gt_actions","history_trajectory","img_shape"]:
        if k in batch[0]: c[k] = torch.stack([b[k] for b in batch])
    if "points" in batch[0]: c["points"] = [b["points"] for b in batch]
    if "token" in batch[0]: c["token"] = [b["token"] for b in batch]
    return c
