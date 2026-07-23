#!/usr/bin/env python3
"""
V6: Two-stage training.
Stage 1: Train on synthetic CA vs synthetic normal (learn CA features)
Stage 2: Fine-tune with real wearable PPG as normal class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import sys
import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.dann.model_v5 import CardiacArrestDetectorV5


class DS(Dataset):
    def __init__(self, ppg, acc, bio, lab, idx=None, aug=False):
        self.ppg = ppg
        self.acc = acc
        self.bio = bio
        self.lab = lab
        self.idx = idx if idx is not None else np.arange(len(lab))
        self.aug = aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        p = self.ppg[ri].copy().astype(np.float32)
        a = self.acc[ri].copy().astype(np.float32)
        b = self.bio[ri].copy().astype(np.float32)
        l = float(self.lab[ri])
        if self.aug:
            rng = np.random.RandomState()
            if rng.random() < 0.3:
                p = np.roll(p, rng.randint(-50, 50))
            if rng.random() < 0.3:
                p = p * rng.uniform(0.8, 1.2)
            if rng.random() < 0.3:
                p = p + rng.normal(0, 0.05, len(p)).astype(np.float32)
            if rng.random() < 0.4:
                start = rng.randint(0, max(1, len(a) - 200))
                dur = min(rng.randint(50, 200), len(a) - start)
                a[start:start+dur] += rng.normal(0, 0.5, (dur, 3)).astype(np.float32)
        return {
            "ppg": torch.tensor(p).unsqueeze(0),
            "accel": torch.tensor(a).permute(1, 0),
            "biodata": torch.tensor(b),
            "label": torch.tensor(l, dtype=torch.float32),
        }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = ROOT / "models/cardiac_arrest_v6"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("V6: TWO-STAGE (Synthetic CA → Fine-tune with Real Normal)", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    sp = np.load(ROOT / "data/processed/synthetic_v5/ppg.npy")
    sa = np.load(ROOT / "data/processed/synthetic_v5/accel.npy")
    sb = pd.read_csv(ROOT / "data/processed/synthetic_v5/biodata.csv").values.astype(np.float32)
    sl = np.load(ROOT / "data/processed/synthetic_v5/labels.npy")

    signals = ROOT / "data/processed/signals"
    rp_list, ra_list, rb_list = [], [], []
    default_bio = np.array([35, .5, 24, .98, 36.8, 2.5, .3, 0, 0, 0, 0, 0, 0, 0, 0, 15], dtype=np.float32)
    mimic_bio = np.array([65, .5, 26, .95, 36.8, 2.0, .3, 0, 0, 0, 0, 0, 0, 0, 0, 10], dtype=np.float32)

    for f in sorted(signals.glob("mmash_*_wear.npy")):
        s = np.load(f).astype(np.float32)
        if s.shape == (1500,) and s.std() > 0.01:
            rp_list.append(s)
            acc = np.zeros((1500, 3), dtype=np.float32); acc[:, 2] = 1.0
            ra_list.append(acc)
            rb_list.append(default_bio.copy())

    for f in sorted(signals.glob("sleepaccel_*_wear.npy")):
        s = np.load(f).astype(np.float32)
        if s.shape == (1500,) and s.std() > 0.01:
            rp_list.append(s)
            acc = np.zeros((1500, 3), dtype=np.float32); acc[:, 2] = 1.0
            ra_list.append(acc)
            rb_list.append(default_bio.copy())

    for f in sorted(signals.glob("mimic_*_wear.npy")):
        s = np.load(f).astype(np.float32)
        if s.shape == (1500,) and s.std() > 0.01:
            rp_list.append(s)
            acc = np.zeros((1500, 3), dtype=np.float32); acc[:, 2] = 1.0
            ra_list.append(acc)
            rb_list.append(mimic_bio.copy())

    rp = np.stack(rp_list)
    ra = np.stack(ra_list)
    rb = np.stack(rb_list)

    ca_mask = sl == 1
    n_ca = int(ca_mask.sum())
    n_norm = int((~ca_mask).sum())
    print(f"  Synthetic: {len(sl)} (CA={n_ca}, Normal={n_norm})", flush=True)
    print(f"  Real normal: {len(rp)}", flush=True)

    # ── Stage 1: Train on synthetic data ──────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("STAGE 1: Synthetic CA vs Synthetic Normal", flush=True)
    print("=" * 70, flush=True)

    pos_w = torch.tensor([min(n_norm / (n_ca + 1e-6), 10.0)], dtype=torch.float32, device=device)

    m = CardiacArrestDetectorV5(n_biodata=16, ppg_dim=128, acc_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    ds_all = DS(sp, sa, sb, sl, aug=True)
    ld_all = DataLoader(ds_all, batch_size=64, shuffle=True, num_workers=0)

    best_loss = float("inf")
    patience = 0
    t0 = time.time()

    for ep in range(50):
        m.train()
        running_loss = 0.0
        n_items = 0
        for batch in ld_all:
            ppg_t = batch["ppg"].to(device)
            acc_t = batch["accel"].to(device)
            bio_t = batch["biodata"].to(device)
            lab_t = batch["label"].to(device)
            optimizer.zero_grad()
            out = m(ppg_t, acc_t, bio_t)
            loss = criterion(out["logit"], lab_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * lab_t.shape[0]
            n_items += lab_t.shape[0]
        avg_loss = running_loss / n_items
        scheduler.step()
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(m.state_dict(), model_dir / "s1_final.pt")
        else:
            patience += 1
        if patience >= 12:
            print(f"  Early stop at epoch {ep+1}", flush=True)
            break
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}: loss={avg_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  S1 training done ({time.time()-t0:.0f}s)", flush=True)

    # Check S1 on real data
    m.load_state_dict(torch.load(model_dir / "s1_final.pt", map_location=device, weights_only=True))
    m.eval()
    rld = DataLoader(DS(rp, ra, rb, np.zeros(len(rp))), 64, num_workers=0)
    rps = []
    with torch.no_grad():
        for batch in rld:
            out_m = m(batch["ppg"].to(device), batch["accel"].to(device), batch["biodata"].to(device))
            rps.extend(out_m["probability"].cpu().numpy())
    rps = np.array(rps)
    print(f"  S1 on real data: P(CA) mean={rps.mean():.4f} max={rps.max():.4f} >0.5: {int((rps > .5).sum())}/{len(rps)}", flush=True)

    # ── Stage 2: Fine-tune with real normal ───────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: Fine-tune with Real Wearable PPG as Normal", flush=True)
    print("=" * 70, flush=True)

    m2 = CardiacArrestDetectorV5(n_biodata=16, ppg_dim=128, acc_dim=64).to(device)
    m2.load_state_dict(torch.load(model_dir / "s1_final.pt", map_location=device, weights_only=True))

    m_ppg = np.concatenate([sp[ca_mask], rp])
    m_acc = np.concatenate([sa[ca_mask], ra])
    m_bio = np.concatenate([sb[ca_mask], rb])
    m_lab = np.concatenate([np.ones(n_ca, dtype=float), np.zeros(len(rp), dtype=float)])

    rng = np.random.RandomState(42)
    si = rng.permutation(len(m_lab))
    m_ppg = m_ppg[si]
    m_acc = m_acc[si]
    m_bio = m_bio[si]
    m_lab = m_lab[si]
    n_tr = int(0.8 * len(m_lab))
    print(f"  Mixed: CA={n_ca}, Real Normal={len(rp)}, Total={len(m_lab)}", flush=True)

    tri = np.arange(n_tr)
    vai = np.arange(n_tr, len(m_lab))
    trld = DataLoader(DS(m_ppg, m_acc, m_bio, m_lab, tri, aug=True), 64, shuffle=True, num_workers=0)
    vrld = DataLoader(DS(m_ppg, m_acc, m_bio, m_lab, vai), 64, num_workers=0)

    n_ca_t = int((m_lab[tri] == 1).sum())
    n_no_t = int((m_lab[tri] == 0).sum())
    pw = torch.tensor([min(n_no_t / (n_ca_t + 1e-6), 10.0)], dtype=torch.float32, device=device)
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer2 = optim.AdamW(m2.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2, eta_min=1e-6)

    best_f1 = 0
    patience = 0
    for ep in range(30):
        m2.train()
        running_loss = 0.0
        n_items = 0
        for batch in trld:
            ppg_t = batch["ppg"].to(device)
            acc_t = batch["accel"].to(device)
            bio_t = batch["biodata"].to(device)
            lab_t = batch["label"].to(device)
            optimizer2.zero_grad()
            out = m2(ppg_t, acc_t, bio_t)
            loss = criterion2(out["logit"], lab_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            optimizer2.step()
            running_loss += loss.item() * lab_t.shape[0]
            n_items += lab_t.shape[0]
        scheduler2.step()

        m2.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in vrld:
                out = m2(batch["ppg"].to(device), batch["accel"].to(device), batch["biodata"].to(device))
                val_probs.extend(out["probability"].cpu().numpy())
                val_labels.extend(batch["label"].numpy())
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        vf1 = f1_score(val_labels, (val_probs > .5).astype(float))
        try:
            vauc = roc_auc_score(val_labels, val_probs)
        except Exception:
            vauc = 0

        if vf1 > best_f1:
            best_f1 = vf1
            patience = 0
            torch.save(m2.state_dict(), model_dir / "s2_best.pt")
            print(f"  Epoch {ep+1:2d}: loss={running_loss/n_items:.4f} val_f1={vf1:.3f} val_auroc={vauc:.3f} *", flush=True)
        else:
            patience += 1
        if patience >= 10:
            print(f"  Early stop at epoch {ep+1}", flush=True)
            break

    # ── Final Evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("FINAL EVALUATION", flush=True)
    print("=" * 70, flush=True)

    m2.load_state_dict(torch.load(model_dir / "s2_best.pt", map_location=device, weights_only=True))
    m2.eval()

    # Real data
    rld = DataLoader(DS(rp, ra, rb, np.zeros(len(rp))), 64, num_workers=0)
    rps = []
    with torch.no_grad():
        for batch in rld:
            out_m = m2(batch["ppg"].to(device), batch["accel"].to(device), batch["biodata"].to(device))
            rps.extend(out_m["probability"].cpu().numpy())
    rps = np.array(rps)

    # Synthetic CA
    cald = DataLoader(DS(sp[ca_mask], sa[ca_mask], sb[ca_mask], np.ones(n_ca)), 64, num_workers=0)
    cps = []
    with torch.no_grad():
        for batch in cald:
            out_m = m2(batch["ppg"].to(device), batch["accel"].to(device), batch["biodata"].to(device))
            cps.extend(out_m["probability"].cpu().numpy())
    cps = np.array(cps)

    print(f"\n  Real normal P(CA):  mean={rps.mean():.4f}  FP={int((rps > .5).sum())}/{len(rps)} ({(rps > .5).mean()*100:.1f}%)", flush=True)
    print(f"  Synthetic CA P(CA): mean={cps.mean():.4f}  TP={int((cps > .5).sum())}/{len(cps)} ({(cps > .5).mean()*100:.1f}%)", flush=True)

    al = np.concatenate([np.zeros(len(rps)), np.ones(len(cps))])
    ap = np.concatenate([rps, cps])
    try:
        auroc = roc_auc_score(al, ap)
        print(f"  AUROC: {auroc:.4f}", flush=True)
    except Exception as e:
        auroc = 0
        print(f"  AUROC: {e}", flush=True)

    # Edge gate
    Xg = np.column_stack([ap, np.concatenate([rb, sb[ca_mask]])])
    rg = RandomForestClassifier(100, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rg.fit(Xg, al)
    rp2 = rg.predict_proba(Xg)[:, 1]
    rf_auc = roc_auc_score(al, rp2)
    rf_f1 = f1_score(al, (rp2 > .5).astype(float))
    print(f"  Edge Gate: AUROC={rf_auc:.4f} F1={rf_f1:.4f}", flush=True)
    with open(model_dir / "edge_gate.pkl", "wb") as f:
        pickle.dump(rg, f)

    # ONNX
    onnx_path = model_dir / "cardiac_arrest_detector_v6.onnx"
    m2.export_onnx(onnx_path, n_biodata=16)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"  ONNX: {onnx_path.stat().st_size / 1024:.0f} KB", flush=True)

    torch.save({"model_state_dict": m2.state_dict(), "config": {"n_biodata": 16, "ppg_dim": 128, "acc_dim": 64}}, model_dir / "best_model.pt")
    with open(model_dir / "cv_results.json", "w") as f:
        json.dump({
            "real_fp": float((rps > .5).mean()),
            "ca_tp": float((cps > .5).mean()),
            "auroc": float(auroc),
            "edge_gate_auroc": float(rf_auc),
            "edge_gate_f1": float(rf_f1),
            "dataset": {"synthetic_ca": n_ca, "real_normal": len(rp)},
        }, f, indent=2)

    print(f"\nDone! Files in {out}", flush=True)


if __name__ == "__main__":
    main()
