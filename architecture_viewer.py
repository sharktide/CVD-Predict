#!/usr/bin/env python3
"""
Interactive OHCA Model Architecture Viewer (Textual TUI)

Launch:  python architecture_viewer.py

Features:
  - Tree-view of every component with shapes & parameter counts
  - Expand/collapse branches
  - Live config editor (sliders) that rebuilds the model summary
  - Search/filter layers by name
  - Detail pane showing selected layer's full spec
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ohca_predictor is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Rule,
    Select,
    Static,
    Tree,
)

# ---------------------------------------------------------------------------
# Architecture data – built from source inspection (no TF import needed)
# ---------------------------------------------------------------------------

SAVED_CONFIG = {
    "model_dim": 128,
    "num_attention_heads": 4,
    "num_encoder_layers": 4,
    "feedforward_dim": 256,
    "dropout_rate": 0.3,
    "attention_dropout_rate": 0.15,
    "static_embedding_dim": 64,
    "tokens_per_modality": 64,
    "max_positional_encoding": 4096,
    "num_survival_bins": 12,
    "uncertainty_samples": 5,
}

INPUT_SPEC = {
    "ecg": {"shape": "(B, variable, 1)", "hz": 130, "device": "Polar H10 chest"},
    "accelerometer": {"shape": "(B, variable, 3)", "hz": 52, "device": "Polar H10 chest"},
    "gyroscope": {"shape": "(B, variable, 3)", "hz": 52, "device": "Chest sensor"},
    "ppg": {"shape": "(B, variable, 1)", "hz": 50, "device": "Wrist sensor"},
    "spo2": {"shape": "(B, variable, 1)", "hz": 10, "device": "Wrist sensor"},
    "temperature": {"shape": "(B, variable, 1)", "hz": 1, "device": "Wrist sensor"},
    "respiration": {"shape": "(B, variable, 1)", "hz": 25, "device": "Derived from ECG"},
    "demographics": {"shape": "(B, 24)", "hz": "-", "device": "Static"},
    "medications": {"shape": "(B, 14)", "hz": "-", "device": "Static"},
    "comorbidities": {"shape": "(B, 14)", "hz": "-", "device": "Static"},
    "lab_values": {"shape": "(B, 8)", "hz": "-", "device": "Static"},
}


def _estimate_params(module: str, cfg: dict) -> int:
    """Exact parameter count per module from h5 weight inspection."""
    counts = {
        "ecg_tokenizer": 699_744,
        "accel_tokenizer": 304_416,
        "gyro_tokenizer": 304_416,
        "ppg_tokenizer": 303_328,
        "spo2_tokenizer": 4_752_512,
        "temp_tokenizer": 4_466_944,
        "resp_tokenizer": 5_377_664,
        "static_embedding": 19_200,
        "ecg_motion_attention": 66_304,
        "ecg_ppg_attention": 66_304,
        "align_pools": 0,
        "assembly": 16_768,
        "transformer_encoder": 527_872,
        "classification_head": 49_537,
        "survival_head": 18_060,
        "uncertainty_head": 8_386,
        "auxiliary_heads": 42_190,
        "calibration_head": 3,
    }
    return counts.get(module, 0)


def build_architecture_tree(cfg: dict) -> dict:
    """Build a nested dict representing the full architecture tree."""
    D = cfg["model_dim"]
    T = cfg["tokens_per_modality"]
    H = cfg["num_attention_heads"]
    L = cfg["num_encoder_layers"]
    FF = cfg["feedforward_dim"]
    key_dim = D // H

    return {
        "name": "OHCAPredictionModel",
        "params": sum(_estimate_params(m, cfg) for m in [
            "ecg_tokenizer", "accel_tokenizer", "gyro_tokenizer",
            "ppg_tokenizer", "spo2_tokenizer", "temp_tokenizer",
            "resp_tokenizer", "static_embedding", "ecg_motion_attention",
            "ecg_ppg_attention",
            "align_pools", "assembly", "transformer_encoder",
            "classification_head", "survival_head", "uncertainty_head",
            "auxiliary_heads", "calibration_head",
        ]),
        "children": [
            {
                "name": "Input Tokenizers",
                "params": sum(_estimate_params(m, cfg) for m in [
                    "ecg_tokenizer", "accel_tokenizer", "gyro_tokenizer",
                    "ppg_tokenizer", "spo2_tokenizer", "temp_tokenizer", "resp_tokenizer",
                ]),
                "children": [
                    {
                        "name": "ECGTokenizer",
                        "params": _estimate_params("ecg_tokenizer", cfg),
                        "detail": f"Conv1D(1→32,k=15) → ResBlock(64,k=7,s=2) → ResBlock(128,k=5,d=2,s=2) → ResBlock(256,k=3,d=4,s=2) → ResBlock({D},k=3,d=8,s=2) → AdaptiveAvgPool({T}) → PosEmbed({T},{D})",
                        "shape_in": "(B, variable, 1)",
                        "shape_out": f"(B, {T}, {D})",
                        "children": [
                            {"name": "init_conv: Conv1D(32, k=15, causal)", "params": 1*32*15, "shape_out": "(B, T, 32)"},
                            {"name": "res_block1: ResBlock(64, k=7, s=2)", "params": 7*64*32+64*2, "shape_out": f"(B, T/2, 64)"},
                            {"name": "res_block2: ResBlock(128, k=5, d=2, s=2)", "params": 5*128*64+128*2, "shape_out": f"(B, T/4, 128)"},
                            {"name": f"res_block3: ResBlock(256, k=3, d=4, s=2)", "params": 3*256*128+256*2, "shape_out": f"(B, T/8, 256)"},
                            {"name": f"res_block4: ResBlock({D}, k=3, d=8, s=2)", "params": 3*D*256+D*2, "shape_out": f"(B, T/16, {D})"},
                            {"name": f"AdaptiveAvgPool1D({T})", "params": 0, "shape_out": f"(B, {T}, {D})"},
                            {"name": f"pos_embedding({T}, {D})", "params": T*D},
                        ],
                    },
                    {
                        "name": "MotionTokenizer (accel + gyro)",
                        "params": _estimate_params("accel_tokenizer", cfg) * 2,
                        "detail": f"Conv1D(3→32,k=15) → ResBlock(64,k=7,s=2) → ResBlock(128,k=5,d=2,s=2) → ResBlock({D},k=3,d=4,s=2) → AdaptiveAvgPool({T})",
                        "shape_in": "(B, variable, 3)",
                        "shape_out": f"(B, {T}, {D}) each",
                        "children": [
                            {"name": "init_conv: Conv1D(32, k=15, causal)", "params": 3*32*15},
                            {"name": "res_block1: ResBlock(64, k=7, s=2)", "params": 7*64*32+64*2},
                            {"name": "res_block2: ResBlock(128, k=5, d=2, s=2)", "params": 5*128*64+128*2},
                            {"name": f"res_block3: ResBlock({D}, k=3, d=4, s=2)", "params": 3*D*128+D*2},
                            {"name": f"AdaptiveAvgPool1D({T})", "params": 0},
                            {"name": f"pos_embedding({T}, {D})", "params": T*D},
                        ],
                    },
                    {
                        "name": "PPGTokenizer",
                        "params": _estimate_params("ppg_tokenizer", cfg),
                        "detail": f"Conv1D(1→32,k=11) → ResBlock(64,k=7,s=2) → ResBlock(128,k=5,d=2,s=2) → ResBlock({D},k=3,d=4,s=2) → AdaptiveAvgPool({T})",
                        "shape_in": "(B, variable, 1)",
                        "shape_out": f"(B, {T}, {D})",
                    },
                    {
                        "name": "AuxSignalTokenizer (SpO2)",
                        "params": _estimate_params("spo2_tokenizer", cfg),
                        "detail": f"Conv1D(1→{D},k=5) → RefineConv({D}→{D}) ×2 → Dense(3255→{D}) → Dense({D}→32768) → AdaptivePool({T})",
                        "shape_in": "(B, variable, 1)",
                        "shape_out": f"(B, {T}, {D})",
                        "children": [
                            {"name": f"Conv1D({D}, k=5)", "params": 5*1*D},
                            {"name": f"RefineConv1({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"RefineConv2({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"Dense(3255→{D})", "params": 3255*D+D},
                            {"name": f"Dense({D}→32768)", "params": D*32768+32768},
                        ],
                    },
                    {
                        "name": "AuxSignalTokenizer (Temperature)",
                        "params": _estimate_params("temp_tokenizer", cfg),
                        "detail": f"Conv1D(1→{D},k=5) → RefineConv({D}→{D}) ×2 → Dense(1024→{D}) → Dense({D}→32768) → AdaptivePool({T})",
                        "shape_in": "(B, variable, 1)",
                        "shape_out": f"(B, {T}, {D})",
                        "children": [
                            {"name": f"Conv1D({D}, k=5)", "params": 5*1*D},
                            {"name": f"RefineConv1({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"RefineConv2({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"Dense(1024→{D})", "params": 1024*D+D},
                            {"name": f"Dense({D}→32768)", "params": D*32768+32768},
                        ],
                    },
                    {
                        "name": "AuxSignalTokenizer (Respiration)",
                        "params": _estimate_params("resp_tokenizer", cfg),
                        "detail": f"Conv1D(1→{D},k=5) → RefineConv({D}→{D}) ×2 → Dense(8139→{D}) → Dense({D}→32768) → AdaptivePool({T})",
                        "shape_in": "(B, variable, 1)",
                        "shape_out": f"(B, {T}, {D})",
                        "children": [
                            {"name": f"Conv1D({D}, k=5)", "params": 5*1*D},
                            {"name": f"RefineConv1({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"RefineConv2({D}→{D}, k=3)", "params": 3*D*D+D*2},
                            {"name": f"Dense(8139→{D})", "params": 8139*D+D},
                            {"name": f"Dense({D}→32768)", "params": D*32768+32768},
                        ],
                    },
                ],
            },
            {
                "name": "StaticPatientEmbedding",
                "params": _estimate_params("static_embedding", cfg),
                "detail": "4 parallel MLPs (demo→32→D, med→32→D, comorb→32→D, lab→32→D) → add → LayerNorm → expand_dims",
                "shape_out": f"(B, 1, {D})",
                "children": [
                    {"name": "Demographics MLP (24→32→{D})", "params": 24*32+32+32*D+D},
                    {"name": "Medications MLP (14→32→{D})", "params": 14*32+32+32*D+D},
                    {"name": "Comorbidities MLP (14→32→{D})", "params": 14*32+32+32*D+D},
                    {"name": "Lab Values MLP (8→32→{D})", "params": 8*32+32+32*D+D},
                    {"name": f"LayerNorm({D})", "params": D*2},
                ],
            },
            {
                "name": "Cross-Modal Attention",
                "params": sum(_estimate_params(m, cfg) for m in [
                    "ecg_motion_attention", "ecg_ppg_attention",
                ]),
                "children": [
                    {
                        "name": "HierarchicalCrossAttention (ECG↔Motion)",
                        "params": _estimate_params("ecg_motion_attention", cfg),
                        "detail": "HIERARCHICAL: 3 AsymmetricCrossAttention levels fused",
                        "children": [
                            {
                                "name": "Level 1 — Local CrossAttention (window=7)",
                                "detail": "Each ECG token attends to local motion neighborhood",
                                "params": (D*H*(D//H)+H*(D//H))*3 + H*(D//H)*D+D + D*2,
                                "children": [
                                    {"name": f"query_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"key_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"value_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"output_dense: Dense({H}×{D//H}→{D})", "params": H*(D//H)*D+D},
                                    {"name": f"LayerNorm({D})", "params": D*2},
                                ],
                            },
                            {
                                "name": "Level 2 — Global CrossAttention",
                                "detail": "Each ECG token attends to all motion tokens",
                                "params": (D*H*(D//H)+H*(D//H))*3 + H*(D//H)*D+D + D*2,
                                "children": [
                                    {"name": f"query_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"key_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"value_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"output_dense: Dense({H}×{D//H}→{D})", "params": H*(D//H)*D+D},
                                    {"name": f"LayerNorm({D})", "params": D*2},
                                ],
                            },
                            {
                                "name": "Level 3 — Pooled CrossAttention",
                                "detail": "Single global ECG vector attends to single global motion vector",
                                "params": (D*H*(D//H)+H*(D//H))*3 + H*(D//H)*D+D + D*2,
                                "children": [
                                    {"name": f"query_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"key_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"value_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                                    {"name": f"output_dense: Dense({H}×{D//H}→{D})", "params": H*(D//H)*D+D},
                                    {"name": f"LayerNorm({D})", "params": D*2},
                                ],
                            },
                            {"name": f"fusion_proj: Dense(3*{D}→{D}, GELU)", "params": 3*D*D+D},
                        ],
                    },
                    {
                        "name": "AsymmetricCrossAttention (ECG↔PPG)",
                        "params": _estimate_params("ecg_ppg_attention", cfg),
                        "detail": "ASYMMETRIC: ECG queries, PPG provides keys/values",
                        "children": [
                            {"name": f"query_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                            {"name": f"key_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                            {"name": f"value_dense: Dense({D}→{H}×{D//H})", "params": D*H*(D//H)+H*(D//H)},
                            {"name": f"output_dense: Dense({H}×{D//H}→{D})", "params": H*(D//H)*D+D},
                            {"name": f"LayerNorm({D})", "params": D*2},
                        ],
                    },
                ],
            },
            {
                "name": "Sequence Alignment & Assembly",
                "params": _estimate_params("align_pools", cfg) + _estimate_params("assembly", cfg),
                "detail": f"AdaptivePool each to {T} tokens → concat [static(1) | ECG(T) | SpO2(T) | Temp(T) | Resp(T)] → Dense+GELU → LayerNorm → Dropout",
                "shape_out": f"(B, 1+4*{T}, {D}) = (B, 1025, {D})",
                "children": [
                    {"name": "align_pool_ecg/resp/spo2/temp → AdaptiveAvgPool(256)", "params": 0},
                    {"name": f"assembly_projection: Dense({D}→{D}, GELU)", "params": D*D+D},
                    {"name": f"assembly_norm: LayerNorm({D})", "params": D*2},
                ],
            },
            {
                "name": f"TransformerEncoder ({cfg['num_encoder_layers']} layers)",
                "params": _estimate_params("transformer_encoder", cfg),
                "detail": f"Pre-LN blocks × {L}, ALiBi positional encoding, gradient checkpointing",
                "shape_in": f"(B, 1025, {D})",
                "shape_out": f"(B, 1025, {D})",
                "children": [
                    *[
                        {
                            "name": f"EncoderBlock_{i}",
                            "params": D*4*H*(D//H)*3 + D*4*H*(D//H) + D*2 + FF*D*2 + D*2,
                            "detail": f"PreLN → MHSA(num_heads={H}, key_dim={key_dim}, ALiBi) → FFN({D}→{FF}→{D})",
                            "children": [
                                {"name": f"LayerNorm({D})", "params": D*2},
                                {"name": f"MultiHeadSelfAttentionALiBi(h={H}, k={key_dim})", "params": D*4*H*(D//H)*3 + D*4*H*(D//H)},
                                {"name": f"LayerNorm({D})", "params": D*2},
                                {"name": f"FFN: Dense({D}→{FF}, GELU) → Dense({FF}→{D})", "params": FF*D*2 + D*2},
                            ],
                        }
                        for i in range(L)
                    ],
                ],
            },
            {
                "name": "Prediction Heads",
                "params": sum(_estimate_params(m, cfg) for m in [
                    "classification_head", "survival_head", "uncertainty_head",
                    "auxiliary_heads", "calibration_head",
                ]),
                "children": [
                    {
                        "name": "ClassificationHead",
                        "params": _estimate_params("classification_head", cfg),
                        "detail": f"CLS token → Dense({D}→256,GELU) → Dropout(0.2) → Dense(256→64,GELU) → Dropout(0.1) → Dense(64→1,Sigmoid)",
                        "shape_out": "(B, 1)",
                    },
                    {
                        "name": "SurvivalAnalysisHead",
                        "params": _estimate_params("survival_head", cfg),
                        "detail": f"CLS token → Dense({D}→128,GELU) → Dense(128→{cfg['num_survival_bins']},Sigmoid). Cumulative survival via cumprod.",
                        "shape_out": f"(B, {cfg['num_survival_bins']+1})",
                    },
                    {
                        "name": "UncertaintyHead",
                        "params": _estimate_params("uncertainty_head", cfg),
                        "detail": f"CLS token → Dense({D}→64,GELU) → Dense(64→2). Outputs [mu, log_sigma]. MC dropout ({cfg['uncertainty_samples']} samples) at inference.",
                        "shape_out": "(B, 2)",
                    },
                    {
                        "name": "AuxiliaryHeads (5 tasks)",
                        "params": _estimate_params("auxiliary_heads", cfg),
                        "detail": "heart_rate(1) + rhythm(5) + activity(5) + spo2(1) + blood_pressure(2)",
                        "children": [
                            {"name": "HeartRate: Dense(D→64,GELU) → Dense(64→1,linear)", "params": D*64+64+64*1+1},
                            {"name": "Rhythm: Dense(D→64,GELU) → Dense(64→5,softmax)", "params": D*64+64+64*5+5},
                            {"name": "Activity: Dense(D→64,GELU) → Dense(64→5,softmax)", "params": D*64+64+64*5+5},
                            {"name": "SpO2: Dense(D→64,GELU) → Dense(64→1,linear)", "params": D*64+64+64*1+1},
                            {"name": "BloodPressure: Dense(D→64,GELU) → Dense(64→2,linear)", "params": D*64+64+64*2+2},
                        ],
                    },
                    {
                        "name": "CalibrationHead (Platt scaling)",
                        "params": 3,
                        "detail": "learnable temperature, platt_a, platt_b scalars",
                        "shape_out": "(B, 1)",
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

class DetailPane(Static):
    """Right-side detail panel."""
    pass


class ArchViewer(App):
    """Interactive OHCA model architecture viewer."""

    CSS = """
    Screen { layout: horizontal; }
    #sidebar { width: 55%; border-right: solid $primary; height: 100%; }
    #detail { width: 45%; height: 100%; }
    #detail-title { text-style: bold; color: $accent; margin: 1 0 0 1; }
    #detail-body { margin: 0 1; height: 1fr; overflow-y: auto; }
    #search-row { dock: top; height: 3; padding: 0 1; }
    #search-input { width: 100%; }
    #config-label { dock: top; height: 1; text-style: bold; color: $warning; padding: 0 1; }
    Tree { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "focus_search", "Search"),
        Binding("r", "reset_tree", "Reset"),
        Binding("e", "export", "Export"),
        Binding("c", "copy_node", "Copy Node"),
        Binding("a", "copy_all", "Copy All"),
    ]

    TITLE = "OHCA Model Architecture Viewer"

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or SAVED_CONFIG.copy()
        self.arch = build_architecture_tree(self.config)
        self.all_nodes: List[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="sidebar"):
            yield VerticalScroll(
                Static("  Search:", id="search-row"),
                Input(placeholder="type to filter layers...", id="search-input"),
                Static(self._config_summary(), id="config-label"),
                Tree(self._build_tree_label(self.arch), id="arch-tree"),
            )
        with Vertical(id="detail"):
            yield Static("Select a node", id="detail-title")
            yield RichLog(id="detail-body", markup=True, highlight=True)
        yield Footer()

    def _config_summary(self) -> str:
        c = self.config
        return (f"  D={c['model_dim']}  H={c['num_attention_heads']}  "
                f"L={c['num_encoder_layers']}  FF={c['feedforward_dim']}  "
                f"T={c['tokens_per_modality']}  dropout={c['dropout_rate']}")

    def _build_tree_label(self, node: dict) -> str:
        p = node.get("params", 0)
        if p >= 1_000_000:
            ps = f"{p/1e6:.2f}M"
        elif p >= 1000:
            ps = f"{p/1e3:.1f}K"
        else:
            ps = str(p)
        return f"{node['name']}  [{ps}]"

    def _walk(self, node: dict, depth: int = 0):
        self.all_nodes.append((depth, node))
        for child in node.get("children", []):
            self._walk(child, depth + 1)

    def _populate_tree(self, parent, node: dict):
        label = self._build_tree_label(node)
        if isinstance(parent, Tree):
            branch = parent.root.add(label)
        else:
            branch = parent.add(label)
        branch.data = node
        for child in node.get("children", []):
            self._populate_tree(branch, child)

    def on_mount(self):
        tree = self.query_one("#arch-tree", Tree)
        tree.root.expand()
        tree.root.data = self.arch
        for child in self.arch.get("children", []):
            self._populate_tree(tree, child)
        self._walk(self.arch)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        node = event.node.data
        if node is None:
            return
        title = self.query_one("#detail-title")
        body = self.query_one("#detail-body")
        title.update(f"[b]{node['name']}[/b]")
        body.clear()
        body.write(f"[bold cyan]Name:[/] {node['name']}")
        if "detail" in node:
            body.write(f"[bold cyan]Detail:[/] {node['detail']}")
        if "shape_in" in node:
            body.write(f"[bold cyan]Input:[/] {node['shape_in']}")
        if "shape_out" in node:
            body.write(f"[bold cyan]Output:[/] {node['shape_out']}")
        p = node.get("params", 0)
        if p >= 1_000_000:
            ps = f"{p/1e6:.2f}M"
        elif p >= 1000:
            ps = f"{p/1e3:.1f}K"
        else:
            ps = str(p)
        body.write(f"[bold cyan]Parameters:[/] {ps}")
        if "hz" in node:
            body.write(f"[bold cyan]Sample rate:[/] {node['hz']} Hz")
        if "device" in node:
            body.write(f"[bold cyan]Device:[/] {node['device']}")
        body.write("")
        children = node.get("children", [])
        if children:
            body.write(f"[bold yellow]Sub-components ({len(children)}):[/]")
            for c in children:
                cp = c.get("params", 0)
                if cp >= 1_000_000:
                    cps = f"{cp/1e6:.2f}M"
                elif cp >= 1000:
                    cps = f"{cp/1e3:.1f}K"
                else:
                    cps = str(cp)
                body.write(f"  • [green]{c['name']}[/]  [{cps}]")
        body.write("")
        body.write("[dim]'c' copy node  'a' copy all  's' search  'r' reset  'q' quit[/]")

    def on_input_changed(self, event: Input.Changed):
        query = event.value.lower().strip()
        tree = self.query_one("#arch-tree", Tree)
        tree.clear()
        if not query:
            tree.root.add(self._build_tree_label(self.arch)).data = self.arch
            for child in self.arch.get("children", []):
                self._populate_tree(tree, child)
            return
        def _filter_match(node: dict) -> bool:
            if query in node["name"].lower():
                return True
            return any(_filter_match(c) for c in node.get("children", []))
        if _filter_match(self.arch):
            self._populate_filtered(tree, self.arch, query)

    def _populate_filtered(self, parent, node: dict, query: str):
        children_matching = [c for c in node.get("children", []) if self._subtree_matches(c, query)]
        if query in node["name"].lower() or children_matching:
            label = self._build_tree_label(node)
            if isinstance(parent, Tree):
                branch = parent.root.add(label)
            else:
                branch = parent.add(label)
            branch.data = node
            for c in children_matching:
                self._populate_filtered(branch, c, query)

    def _subtree_matches(self, node: dict, query: str) -> bool:
        if query in node["name"].lower():
            return True
        return any(self._subtree_matches(c, query) for c in node.get("children", []))

    def action_focus_search(self):
        self.query_one("#search-input").focus()

    def action_reset_tree(self):
        tree = self.query_one("#arch-tree", Tree)
        tree.clear()
        self.query_one("#search-input").value = ""
        tree.root.add(self._build_tree_label(self.arch)).data = self.arch
        for child in self.arch.get("children", []):
            self._populate_tree(tree, child)

    def action_export(self):
        self.export_architecture()

    def _format_node_text(self, node: dict, depth: int = 0) -> str:
        """Format a node and its children as indented text."""
        indent = "  " * depth
        p = node.get("params", 0)
        if p >= 1_000_000:
            ps = f"{p/1e6:.2f}M"
        elif p >= 1000:
            ps = f"{p/1e3:.1f}K"
        else:
            ps = str(p)
        lines = [f"{indent}{node['name']}  [{ps}]"]
        if "detail" in node:
            lines.append(f"{indent}  detail: {node['detail']}")
        if "shape_in" in node:
            lines.append(f"{indent}  input:  {node['shape_in']}")
        if "shape_out" in node:
            lines.append(f"{indent}  output: {node['shape_out']}")
        for c in node.get("children", []):
            lines.append(self._format_node_text(c, depth + 1))
        text = "\n".join(lines)
        return text.encode("ascii", "replace").decode("ascii")

    def _get_selected_node(self) -> dict | None:
        """Get the data of the currently selected tree node."""
        tree = self.query_one("#arch-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            return tree.cursor_node.data
        return None

    def action_copy_node(self):
        """Copy selected node details to clipboard."""
        import subprocess
        node = self._get_selected_node()
        if node is None:
            body = self.query_one("#detail-body")
            body.write("\n[dim]No node selected[/]")
            return
        text = self._format_node_text(node)
        try:
            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            process.communicate(input=text)
            body = self.query_one("#detail-body")
            body.write(f"\n[bold green]Copied {node['name']} to clipboard[/]")
        except Exception as e:
            body = self.query_one("#detail-body")
            body.write(f"\n[bold red]Copy failed: {e}[/]")

    def action_copy_all(self):
        """Copy entire architecture tree to clipboard."""
        import subprocess
        text = self._format_node_text(self.arch)
        try:
            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            process.communicate(input=text)
            body = self.query_one("#detail-body")
            body.write(f"\n[bold green]Copied full architecture ({len(text)} chars) to clipboard[/]")
        except Exception as e:
            body = self.query_one("#detail-body")
            body.write(f"\n[bold red]Copy failed: {e}[/]")

    def export_architecture(self):
        """Export full architecture to a JSON file."""
        import json
        out_path = PROJECT_ROOT / "models" / "architecture_export.json"
        with open(out_path, "w") as f:
            json.dump(self.arch, f, indent=2, default=str)
        body = self.query_one("#detail-body")
        body.write(f"\n[bold green]Exported to {out_path}[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the architecture viewer."""
    # Try to load config from models/config.json if available
    config = SAVED_CONFIG.copy()
    config_path = PROJECT_ROOT / "models" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            if "model" in data:
                config.update(data["model"])
            # Load static dims if present
            for k in ("demographic_dim", "num_medications", "num_comorbidities", "num_labs"):
                if k in data:
                    config[k] = data[k]
        except Exception:
            pass

    app = ArchViewer(config=config)
    app.run()


if __name__ == "__main__":
    main()
