"""
V5 Cardiac Arrest Detection Model - Multi-Branch Wrist PPG Architecture

Architecture:
- Branch A: PPG ResNet (1D CNN with residual connections)
- Branch B: ACC CNN (3-axis accelerometer encoder)
- Cross-Attention Fusion: ACC attends PPG to suppress motion artifacts
- Edge Decision Gate: Lightweight Random Forest on P(ischemia) + biodata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# ── Residual Blocks ──────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Residual block with 1D convolution."""

    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.act(out)


class DownsampleBlock(nn.Module):
    """Conv + ResBlock + MaxPool for downsampling."""

    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.res = ResBlock1D(out_ch, kernel_size=5, dropout=dropout)
        self.pool = nn.MaxPool1d(2)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.res(x)
        x = self.pool(x)
        return x


# ── Branch A: PPG Encoder ───────────────────────────────────────────────────

class PPGBranch(nn.Module):
    """1D ResNet for PPG waveform encoding."""

    def __init__(self, latent_dim=128):
        super().__init__()
        # 1500 → 750 → 375 → 187 → 93 → 46
        self.blocks = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7), nn.BatchNorm1d(32), nn.GELU(),
            ResBlock1D(32, 15, dropout=0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            ResBlock1D(64, 7, dropout=0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            ResBlock1D(128, 5, dropout=0.15),
            nn.MaxPool1d(2),

            nn.Conv1d(128, latent_dim, 3, stride=2, padding=1), nn.BatchNorm1d(latent_dim), nn.GELU(),
            ResBlock1D(latent_dim, 3, dropout=0.15),
        )

    def forward(self, x):
        return self.blocks(x)  # (batch, latent_dim, ~46)


# ── Branch B: ACC Encoder ───────────────────────────────────────────────────

class ACCBranch(nn.Module):
    """1D CNN for 3-axis accelerometer encoding."""

    def __init__(self, latent_dim=64):
        super().__init__()
        # 1500 → 750 → 375 → 187 → 93
        self.blocks = nn.Sequential(
            nn.Conv1d(3, 32, 15, stride=2, padding=7), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, latent_dim, 5, stride=2, padding=2), nn.BatchNorm1d(latent_dim), nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, 3, padding=1), nn.BatchNorm1d(latent_dim), nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.blocks(x)  # (batch, latent_dim, ~93)


# ── Cross-Attention Fusion ──────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    ACC branch attends to PPG branch to suppress motion artifacts.
    ACC tells PPG which temporal segments are distorted by movement.
    """
    
    def __init__(self, ppg_dim=128, acc_dim=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = ppg_dim // n_heads
        
        # Project ACC to PPG dimension for attention
        self.acc_proj = nn.Linear(acc_dim, ppg_dim)
        
        # Multi-head cross-attention
        self.q_proj = nn.Linear(ppg_dim, ppg_dim)
        self.k_proj = nn.Linear(ppg_dim, ppg_dim)
        self.v_proj = nn.Linear(ppg_dim, ppg_dim)
        self.out_proj = nn.Linear(ppg_dim, ppg_dim)
        
        self.norm = nn.LayerNorm(ppg_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Motion gating: ACC signal gates PPG features
        self.gate = nn.Sequential(
            nn.Linear(acc_dim, ppg_dim),
            nn.Sigmoid(),
        )

    def forward(self, ppg_features, acc_features):
        """
        ppg_features: (batch, ppg_dim, ppg_seq) 
        acc_features: (batch, acc_dim, acc_seq)
        """
        B, Dp, Sp = ppg_features.shape
        _, Da, Sa = acc_features.shape
        
        # Pool ACC sequence to single vector for gating
        acc_pooled = acc_features.mean(dim=2)  # (batch, acc_dim)
        
        # Compute motion gate: which PPG segments are trustworthy
        gate = self.gate(acc_pooled).unsqueeze(-1)  # (batch, ppg_dim, 1)
        
        # Reshape for attention
        ppg_2d = ppg_features.permute(0, 2, 1)  # (batch, Sp, ppg_dim)
        acc_2d = acc_features.permute(0, 2, 1)  # (batch, Sa, acc_dim)
        
        # Project ACC to PPG dim
        acc_proj = self.acc_proj(acc_2d)  # (batch, Sa, ppg_dim)
        
        # Cross-attention: Q=PPG, K=V=ACC
        Q = self.q_proj(ppg_2d)  # (batch, Sp, ppg_dim)
        K = self.k_proj(acc_proj)  # (batch, Sa, ppg_dim)
        V = self.v_proj(acc_proj)  # (batch, Sa, ppg_dim)
        
        # Reshape for multi-head
        Q = Q.view(B, Sp, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, Sp, head_dim)
        K = K.view(B, Sa, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Sa, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        context = torch.matmul(attn, V)  # (B, heads, Sp, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, Sp, -1)  # (B, Sp, ppg_dim)
        context = self.out_proj(context)  # (B, Sp, ppg_dim)
        
        # Reshape back
        context = context.permute(0, 2, 1)  # (B, ppg_dim, Sp)
        
        # Apply motion gate: suppress PPG features where ACC shows high motion
        fused = ppg_features * gate + context * (1 - gate)
        fused = self.norm(fused.permute(0, 2, 1)).permute(0, 2, 1)
        
        return fused, attn


# ── Biodata Encoder ─────────────────────────────────────────────────────────

class BiodataEncoder(nn.Module):
    """Encode static patient biodata (age, sex, comorbidities, etc.)."""

    def __init__(self, n_features, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# ── Full V5 Model ───────────────────────────────────────────────────────────

class CardiacArrestDetectorV5(nn.Module):
    """
    Multi-branch wrist PPG cardiac arrest detector.
    
    Inputs:
        ppg: (batch, 1, 1500) - wrist PPG @ 25Hz
        accel: (batch, 3, 1500) - 3-axis accelerometer @ 25Hz
        biodata: (batch, n_biodata) - patient metadata
    
    Outputs:
        logit: (batch,) - raw logit for cardiac arrest
        probability: (batch,) - sigmoid probability
        ppg_latent: (batch, 128) - PPG encoding
        motion_gate: (batch, 128) - which PPG features are gated by ACC
        attention_weights: (batch, heads, ppg_seq, acc_seq) - cross-attention
    """

    def __init__(self, n_biodata=16, ppg_dim=128, acc_dim=64, latent_dim=128):
        super().__init__()
        
        self.ppg_branch = PPGBranch(latent_dim=ppg_dim)
        self.acc_branch = ACCBranch(latent_dim=acc_dim)
        self.cross_attention = CrossAttentionFusion(ppg_dim=ppg_dim, acc_dim=acc_dim)
        self.biodata_encoder = BiodataEncoder(n_biodata, latent_dim=32)
        
        # Temporal pooling after cross-attention
        self.ppg_pool = nn.AdaptiveAvgPool1d(1)
        self.acc_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion classifier
        fusion_dim = ppg_dim + acc_dim + 32  # PPG + ACC + biodata
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
        self._ppg_dim = ppg_dim
        self._acc_dim = acc_dim
        self._n_biodata = n_biodata

    def forward(self, ppg, accel, biodata):
        # Branch A: PPG
        ppg_feat = self.ppg_branch(ppg)  # (B, ppg_dim, seq)
        
        # Branch B: ACC
        acc_feat = self.acc_branch(accel)  # (B, acc_dim, seq)
        
        # Cross-attention fusion
        fused, attn_weights = self.cross_attention(ppg_feat, acc_feat)
        
        # Pool
        ppg_pooled = self.ppg_pool(fused).squeeze(-1)  # (B, ppg_dim)
        acc_pooled = self.acc_pool(acc_feat).squeeze(-1)  # (B, acc_dim)
        
        # Biodata
        bio_feat = self.biodata_encoder(biodata)  # (B, 32)
        
        # Fuse all
        combined = torch.cat([ppg_pooled, acc_pooled, bio_feat], dim=1)
        logit = self.classifier(combined).squeeze(-1)
        prob = torch.sigmoid(logit)
        
        return {
            "logit": logit,
            "probability": prob,
            "ppg_latent": ppg_pooled,
            "motion_gate": ppg_pooled,  # Placeholder for analysis
            "attention_weights": attn_weights,
        }

    def export_onnx(self, path, n_biodata=16):
        """Export to ONNX for edge deployment."""
        import copy
        self.eval()
        cpu_model = copy.deepcopy(self).cpu()
        cpu_model.eval()
        
        dummy_ppg = torch.randn(1, 1, 1500)
        dummy_accel = torch.randn(1, 3, 1500)
        dummy_biodata = torch.randn(1, n_biodata)
        
        torch.onnx.export(
            cpu_model,
            (dummy_ppg, dummy_accel, dummy_biodata),
            str(path),
            input_names=["ppg", "accel", "biodata"],
            output_names=["logit", "probability", "ppg_latent"],
            opset_version=18,
        )
        print(f"Model exported to {path}")

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
