"""
graph/ — Predictive Coding sur graphes arbitraires avec gestion des cycles.

Ce package implémente :
- pc_gnn.py          : Extension du PC aux graphes arbitraires (PC-GNN)
- phase_sync.py      : Condition de convergence de type Kuramoto pour les cycles
- message_passing.py : Passage de messages avec historique de visite anti-cycle
"""

from graph.phase_sync import (
    kuramoto_condition,
    phase_consistency_check,
    compute_order_parameter,
)
from graph.message_passing import MessagePassingPC
from graph.pc_gnn import PCGNN

__all__ = [
    "kuramoto_condition",
    "phase_consistency_check",
    "compute_order_parameter",
    "MessagePassingPC",
    "PCGNN",
]
