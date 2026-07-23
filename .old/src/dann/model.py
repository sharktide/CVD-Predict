"""
Phase 3: Domain-Adversarial Neural Network (DANN) with Multi-Head Transformer

Architecture:
1. Shared Feature Encoder (E): 1D CNN → Transformer → latent space
2. Domain Classifier (D): Feed-forward with Gradient Reversal Layer (GRL)
3. Temporal Predictor (P): Multi-head transformer with:
   - Head 1: Event Type Classification [Healthy, General Deterioration, Cardiac Arrest]
   - Head 2: Time-to-Event Regression (conditional on Head 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ── Gradient Reversal Layer ───────────────────────────────────────────────────

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


# ── Shared Feature Encoder (E) ───────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """1D convolution block with batch norm and optional residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.has_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.has_residual:
            out = out + x
        return out


class SharedFeatureEncoder(nn.Module):
    """
    Shared Feature Encoder (E) for domain-invariant feature extraction.

    Input: PPG segment (batch, 1, seq_len) or feature vector (batch, n_features)
    Output: Latent representation (batch, latent_dim)
    """

    def __init__(self, input_dim=1500, latent_dim=128, n_features=115):
        super().__init__()

        # 1D CNN for raw PPG input
        self.cnn_encoder = nn.Sequential(
            ConvBlock1D(1, 32, kernel_size=7, stride=2),
            ConvBlock1D(32, 64, kernel_size=5, stride=2),
            ConvBlock1D(64, 128, kernel_size=3, stride=2),
            ConvBlock1D(128, 128, kernel_size=3, stride=2),
        )

        # Adaptive pooling to fixed length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Feature branch (for pre-extracted features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Final projection to latent space
        self.projection = nn.Sequential(
            nn.Linear(128 * 16, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.latent_dim = latent_dim

    def forward(self, x, features=None):
        """
        Args:
            x: Raw PPG signal (batch, 1, seq_len) or flattened PPG (batch, seq_len)
            features: Pre-extracted feature vector (batch, n_features), optional
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # CNN encoding
        cnn_out = self.cnn_encoder(x)  # (batch, 128, L)
        cnn_out = self.adaptive_pool(cnn_out)  # (batch, 128, 16)

        # Transformer
        transformer_in = cnn_out.permute(0, 2, 1)  # (batch, 16, 128)
        transformer_out = self.transformer(transformer_in)  # (batch, 16, 128)

        # Flatten
        cnn_flat = transformer_out.reshape(transformer_out.size(0), -1)  # (batch, 128*16)

        # Feature branch (if features provided)
        if features is not None:
            feat_out = self.feature_encoder(features)  # (batch, 128)
            # Combine: repeat features to match transformer output length
            feat_expanded = feat_out.unsqueeze(1).expand(-1, 16, -1)  # (batch, 16, 128)
            feat_flat = feat_expanded.reshape(feat_out.size(0), -1)  # (batch, 128*16)
            cnn_flat = cnn_flat + feat_flat  # Simple addition fusion

        # Project to latent space
        latent = self.projection(cnn_flat)  # (batch, latent_dim)

        return latent


# ── Domain Classifier (D) ────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Domain Classifier (D) for distinguishing ICU vs Wearable domains.

    Uses Gradient Reversal Layer to force domain-invariant features.
    """

    def __init__(self, input_dim=128, hidden_dim=64, n_domains=2):
        super().__init__()

        self.grl = GradientReversalLayer()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_domains),
        )

    def forward(self, x, alpha=1.0):
        # Apply gradient reversal
        x_reversed = self.grl(x, alpha)
        return self.classifier(x_reversed)


# ── Temporal Predictor (P) ───────────────────────────────────────────────────

class TemporalPredictor(nn.Module):
    """
    Temporal Predictor (P) with multi-head output.

    Head 1: Event Type Classification [Healthy, General Deterioration, Cardiac Arrest]
    Head 2: Time-to-Event Regression (conditional on Head 1 classification)
    """

    def __init__(self, input_dim=128, hidden_dim=128, n_event_types=3):
        super().__init__()

        # Shared temporal processing
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Head 1: Event Type Classification
        # Classes: 0=Healthy, 1=General Deterioration (AKI, HF, RF, Sepsis), 2=Cardiac Arrest
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_event_types),
        )

        # Head 2: Time-to-Event Regression (conditional)
        # Predicts hours until event (0-24h), only meaningful for deteriorating patients
        self.time_regressor = nn.Sequential(
            nn.Linear(hidden_dim + n_event_types, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output 0-1, scale to 0-24 hours
        )

        self.n_event_types = n_event_types

    def forward(self, x):
        """
        Args:
            x: Latent features from Shared Feature Encoder (batch, latent_dim)

        Returns:
            event_logits: Classification logits (batch, n_event_types)
            time_to_event: Predicted time-to-event in hours (batch, 1), scaled 0-24
        """
        # Shared encoding
        h = self.temporal_encoder(x)

        # Head 1: Event classification
        event_logits = self.event_classifier(h)

        # Head 2: Time-to-event (conditional on event classification)
        # Concatenate event probabilities with features
        event_probs = F.softmax(event_logits, dim=-1)
        time_input = torch.cat([h, event_probs], dim=-1)
        time_raw = self.time_regressor(time_input) * 24.0  # Scale to 0-24 hours

        return event_logits, time_raw


# ── Full DANN Model ──────────────────────────────────────────────────────────

class DANN_CardiacArrest(nn.Module):
    """
    Full Domain-Adversarial Neural Network for Cardiac Arrest Prediction.

    Combines:
    1. Shared Feature Encoder (E)
    2. Domain Classifier (D) with GRL
    3. Temporal Predictor (P) with multi-head output
    """

    def __init__(self, input_dim=1500, latent_dim=128, n_features=115, n_event_types=3):
        super().__init__()

        self.encoder = SharedFeatureEncoder(
            input_dim=input_dim, latent_dim=latent_dim, n_features=n_features
        )
        self.domain_classifier = DomainClassifier(
            input_dim=latent_dim, hidden_dim=64, n_domains=2
        )
        self.predictor = TemporalPredictor(
            input_dim=latent_dim, hidden_dim=128, n_event_types=n_event_types
        )

        self.latent_dim = latent_dim
        self.n_event_types = n_event_types

    def forward(self, x, features=None, alpha=1.0):
        """
        Args:
            x: Raw PPG signal (batch, 1, seq_len)
            features: Pre-extracted features (batch, n_features), optional
            alpha: GRL trade-off parameter

        Returns:
            latent: Domain-invariant latent features
            domain_logits: Domain classification logits
            event_logits: Event type classification logits
            time_to_event: Predicted time-to-event (hours)
        """
        # Shared encoding
        latent = self.encoder(x, features)

        # Domain classification (with GRL)
        domain_logits = self.domain_classifier(latent, alpha)

        # Event prediction
        event_logits, time_to_event = self.predictor(latent)

        return {
            "latent": latent,
            "domain_logits": domain_logits,
            "event_logits": event_logits,
            "time_to_event": time_to_event,
        }


# ── Loss Functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DANNLoss(nn.Module):
    """
    Combined loss for DANN training.

    L_total = L_arrest_prediction - λ * L_domain_classification + γ * L_reconstruction
    """

    def __init__(self, event_weights=None, lambda_domain=1.0, gamma_time=1.0):
        super().__init__()

        # Event classification loss (Focal Loss for class imbalance)
        if event_weights is not None:
            event_weights = torch.tensor(event_weights, dtype=torch.float32)
        self.event_loss_fn = FocalLoss(alpha=event_weights, gamma=2.0)

        # Domain classification loss
        self.domain_loss_fn = nn.CrossEntropyLoss()

        # Time-to-event regression loss (only for non-healthy samples)
        self.time_loss_fn = nn.MSELoss(reduction='none')

        self.lambda_domain = lambda_domain
        self.gamma_time = gamma_time

    def forward(self, outputs, event_labels, domain_labels, time_to_event_labels,
                is_healthy_mask=None):
        """
        Args:
            outputs: Model output dict
            event_labels: Event type labels (batch,) - 0=Healthy, 1=General, 2=Arrest
            domain_labels: Domain labels (batch,) - 0=Clinical, 1=Wearable
            time_to_event_labels: Time-to-event labels (batch, 1) in hours
            is_healthy_mask: Boolean mask for healthy samples (batch,)
        """
        # 1. Event classification loss
        event_loss = self.event_loss_fn(outputs["event_logits"], event_labels)

        # 2. Domain classification loss
        domain_loss = self.domain_loss_fn(outputs["domain_logits"], domain_labels)

        # 3. Time-to-event regression loss (only for non-healthy samples)
        if is_healthy_mask is not None:
            # Mask: only compute time loss for non-healthy samples
            non_healthy_mask = ~is_healthy_mask
            if non_healthy_mask.any():
                time_pred = outputs["time_to_event"][non_healthy_mask]
                time_true = time_to_event_labels[non_healthy_mask]
                time_loss = self.time_loss_fn(time_pred, time_true).mean()
            else:
                time_loss = torch.tensor(0.0, device=event_loss.device)
        else:
            time_loss = self.time_loss_fn(outputs["time_to_event"], time_to_event_labels).mean()

        # Combined loss (domain loss is added, not subtracted, to prevent negative dominance)
        total_loss = event_loss + self.gamma_time * time_loss + self.lambda_domain * domain_loss

        return {
            "total_loss": total_loss,
            "event_loss": event_loss,
            "domain_loss": domain_loss,
            "time_loss": time_loss,
        }


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_dann_model(config=None):
    """Build DANN model with default or custom configuration."""
    if config is None:
        config = {
            "input_dim": 1500,      # 60 seconds at 25 Hz
            "latent_dim": 128,      # Latent space dimension
            "n_features": 115,      # Number of pre-extracted features
            "n_event_types": 3,     # Healthy, General Deterioration, Cardiac Arrest
        }

    model = DANN_CardiacArrest(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        n_features=config["n_features"],
        n_event_types=config["n_event_types"],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"DANN Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Domain Classifier: {sum(p.numel() for p in model.domain_classifier.parameters()):,}")
    print(f"  Predictor: {sum(p.numel() for p in model.predictor.parameters()):,}")

    return model


if __name__ == "__main__":
    # Test model
    model = build_dann_model()

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 1500)
    features = torch.randn(batch_size, 115)

    outputs = model(x, features, alpha=1.0)

    print(f"\nForward pass test:")
    print(f"  Input: {x.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Latent: {outputs['latent'].shape}")
    print(f"  Domain logits: {outputs['domain_logits'].shape}")
    print(f"  Event logits: {outputs['event_logits'].shape}")
    print(f"  Time-to-event: {outputs['time_to_event'].shape}")
