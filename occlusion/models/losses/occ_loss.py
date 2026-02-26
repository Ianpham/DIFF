"""Loss functions: OccupancyLoss (CE+Lovász), DiffusionLoss, DecorrelationLoss, CombinedLoss."""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Optional

def lovasz_grad(gt_sorted):
    p = len(gt_sorted); gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1-gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection/union
    if p > 1: jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0: return probas*0.0
    C = probas.shape[1]; losses = []
    class_to_sum = list(range(C)) if classes=="all" else torch.unique(labels)
    for c in class_to_sum:
        if c == 0: continue
        fg = (labels==c).float()
        if classes=="present" and fg.sum()==0: continue
        errors = (fg - (1.0-probas[:,c])).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device, requires_grad=True)

class OccupancyLoss(nn.Module):
    def __init__(self, num_classes=17, empty_label=0, ce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.ce_w = ce_weight; self.lov_w = lovasz_weight
        weights = torch.ones(num_classes); weights[empty_label] = 0.05
        self.register_buffer("class_weights", weights)
    def forward(self, pred, target, mask=None):
        C = pred.shape[-1]; pf = pred.reshape(-1,C); tf = target.reshape(-1).long()
        if mask is not None: m = mask.reshape(-1).bool(); pf = pf[m]; tf = tf[m]
        ce = F.cross_entropy(pf, tf, weight=self.class_weights, ignore_index=255)
        lov = lovasz_softmax_flat(F.softmax(pf, 1), tf)
        return {"ce_loss": ce, "lovasz_loss": lov, "total": self.ce_w*ce + self.lov_w*lov}

class DiffusionLoss(nn.Module):
    def forward(self, pred, target): return F.mse_loss(pred, target)

class DecorrelationLoss(nn.Module):
    def forward(self, rep):
        B, D = rep.shape
        normed = (rep - rep.mean(0, keepdim=True))/(rep.var(0, keepdim=True).sqrt()+1e-8)
        corr = (normed.T @ normed)/B
        mask = ~torch.eye(D, dtype=torch.bool, device=rep.device)
        return (corr[mask]**2).mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=17, loss_weights=None):
        super().__init__()
        self.occ = OccupancyLoss(num_classes); self.diff = DiffusionLoss(); self.decorr = DecorrelationLoss()
        self.w = loss_weights or dict(occ_ce=1.0, occ_lovasz=1.0, diffusion=1.0, decorrelation=0.02)
    def forward(self, occ_pred=None, occ_target=None, noise_pred=None, noise_target=None, fused_repr=None, occ_mask=None):
        losses = {}; dev = next((t.device for t in [occ_pred, noise_pred] if t is not None), torch.device("cpu"))
        total = torch.tensor(0.0, device=dev)
        if occ_pred is not None and occ_target is not None:
            o = self.occ(occ_pred, occ_target, occ_mask)
            for k in ["occ_ce","occ_lovasz"]:
                mk = k.replace("occ_",""); losses[k] = o.get(f"{mk}_loss", o.get(k, torch.tensor(0.0)))
            if "occ_ce" in self.w: total = total + self.w["occ_ce"]*o["ce_loss"]
            if "occ_lovasz" in self.w: total = total + self.w["occ_lovasz"]*o["lovasz_loss"]
        if noise_pred is not None and noise_target is not None:
            d = self.diff(noise_pred, noise_target); losses["diffusion"] = d
            if "diffusion" in self.w: total = total + self.w["diffusion"]*d
        if fused_repr is not None:
            dc = self.decorr(fused_repr); losses["decorrelation"] = dc
            if "decorrelation" in self.w: total = total + self.w["decorrelation"]*dc
        losses["total"] = total; return losses
