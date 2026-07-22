"""
Cardiac Arrest Detection Model - Deployable Architecture

Architecture:
- 1D CNN for raw PPG waveform feature extraction
- Transformer encoder for temporal modeling
- Feature encoder for pre-extracted features (HRV, morphology, etc.)
- Binary classifier: Cardiac Arrest vs Normal

Deployment:
- ONNX export for edge deployment
- Preprocessing pipeline included
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class ConvBlock1D(nn.Module):
    """1D convolution block with batch norm and GELU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class WaveformEncoder(nn.Module):
    """1D CNN encoder for raw PPG waveform (1500 samples @ 25Hz = 60s)."""

    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Progressive downsampling: 1500 -> 750 -> 375 -> 188 -> 94 -> 47
        self.blocks = nn.Sequential(
            ConvBlock1D(1, 32, kernel_size=7, stride=2, dropout=0.1),
            SEBlock(32),
            ConvBlock1D(32, 64, kernel_size=5, stride=2, dropout=0.1),
            SEBlock(64),
            ConvBlock1D(64, 128, kernel_size=5, stride=2, dropout=0.15),
            SEBlock(128),
            ConvBlock1D(128, 128, kernel_size=3, stride=2, dropout=0.15),
            SEBlock(128),
            ConvBlock1D(128, latent_dim, kernel_size=3, stride=2, dropout=0.2),
        )

    def forward(self, x):
        # x: (batch, 1, 1500)
        return self.blocks(x)  # (batch, latent_dim, ~47)


class TemporalEncoder(nn.Module):
    """Transformer encoder for temporal modeling over CNN features."""

    def __init__(self, latent_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, latent_dim, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, latent_dim)
        x = self.transformer(x)  # (batch, seq_len, latent_dim)
        x = x.permute(0, 2, 1)  # (batch, latent_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, latent_dim)
        return x


class FeatureEncoder(nn.Module):
    """MLP encoder for pre-extracted features (HRV, morphology, etc.)."""

    def __init__(self, n_features, latent_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CardiacArrestDetector(nn.Module):
    """
    Deployable cardiac arrest detection model.
    
    Inputs:
        - ppg: Raw PPG waveform (batch, 1, 1500) @ 25Hz
        - features: Pre-extracted features (batch, n_features)
    
    Outputs:
        - logit: Binary logit for cardiac arrest
        - probability: Sigmoid probability
        - latent: 128-d latent representation
    """

    def __init__(self, n_features=40, latent_dim=128):
        super().__init__()
        
        self.waveform_encoder = WaveformEncoder(latent_dim)
        self.temporal_encoder = TemporalEncoder(latent_dim)
        self.feature_encoder = FeatureEncoder(n_features, latent_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Binary classifier
        self.classifier = nn.Linear(128, 1)
        
        self._latent_dim = latent_dim

    def forward(self, ppg, features):
        # Encode PPG waveform
        wave_latent = self.waveform_encoder(ppg)  # (batch, latent_dim, seq_len)
        wave_latent = self.temporal_encoder(wave_latent)  # (batch, latent_dim)
        
        # Encode features
        feat_latent = self.feature_encoder(features)  # (batch, latent_dim)
        
        # Fuse
        combined = torch.cat([wave_latent, feat_latent], dim=1)  # (batch, latent_dim*2)
        fused = self.fusion(combined)  # (batch, 128)
        
        # Classify
        logit = self.classifier(fused)  # (batch, 1)
        prob = torch.sigmoid(logit)
        
        return {
            "logit": logit.squeeze(-1),
            "probability": prob.squeeze(-1),
            "latent": fused,
        }

    @property
    def latent_dim(self):
        return self._latent_dim

    def export_onnx(self, path, n_features=40):
        """Export model to ONNX format."""
        import copy
        self.eval()
        cpu_model = copy.deepcopy(self).cpu()
        cpu_model.eval()
        dummy_ppg = torch.randn(1, 1, 1500)
        dummy_feat = torch.randn(1, n_features)
        
        torch.onnx.export(
            cpu_model,
            (dummy_ppg, dummy_feat),
            str(path),
            input_names=["ppg", "features"],
            output_names=["logit", "probability", "latent"],
            opset_version=18,
        )
        print(f"Model exported to {path}")


def build_model(config: dict) -> CardiacArrestDetector:
    """Build model from config dict."""
    return CardiacArrestDetector(
        n_features=config.get("n_features", 40),
        latent_dim=config.get("latent_dim", 128),
    )
