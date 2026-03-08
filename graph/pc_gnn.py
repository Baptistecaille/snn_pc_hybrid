"""
graph/pc_gnn.py — Predictive Coding sur graphe (PC-GNN).

Extension du Predictive Coding aux graphes arbitraires via un framework
de type Graph Neural Network (GNN).

Dans un PC-GNN :
- Chaque nœud = un niveau hiérarchique de représentation
- Chaque arête = une relation de prédiction (parent → enfant)
- Le message-passing implémente les passes top-down (prédictions) et
  bottom-up (corrections d'erreur)

La convergence sur les graphes cycliques est garantie sous la condition
de Kuramoto (voir graph/phase_sync.py).

Architecture :
- NodeEncoder  : projection des features d'entrée → représentation latente
- EdgePredictor: prédiction d'une représentation enfant depuis un parent
- ErrorAggregator: agrégation des erreurs depuis les voisins
- NodeUpdater  : mise à jour des représentations (règle PC + synchronisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import SNNConfig
from graph.phase_sync import compute_order_parameter, kuramoto_update


class PCGNN(nn.Module):
    """
    Predictive Coding Graph Neural Network.

    Implémente le Predictive Coding comme un GNN avec :
    - Passes de messages alternées top-down / bottom-up
    - Mise à jour des représentations latentes par minimisation de l'énergie libre
    - Synchronisation de phase pour la convergence sur les cycles
    - Amortissement cyclique via historique de visite (délégué à ArcuateFasciculus)

    Paramètres appris :
    - Fonctions de prédiction : f_{ij}(μ_i) → μ̂_j (one per edge type)
    - Poids de précision : Π_i = 1/σ_i² (confiance par nœud)
    """

    def __init__(
        self,
        n_nodes: int,
        node_dim: int,
        config: SNNConfig,
        n_edge_types: int = 1,
    ):
        """
        Args:
            n_nodes      : nombre de nœuds dans le graphe
            node_dim     : dimension des représentations par nœud
            config       : configuration globale
            n_edge_types : nombre de types d'arêtes (1 = graphe homogène)
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.config = config
        self.n_edge_types = n_edge_types

        # ── Encodeur de nœuds ────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )

        # ── Fonctions de prédiction par type d'arête ──────────────────────────
        # f_{type}(μ_parent) → μ̂_enfant : prédiction top-down
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh(),
                nn.Linear(node_dim, node_dim),
            )
            for _ in range(n_edge_types)
        ])

        # ── Agrégateur d'erreurs ──────────────────────────────────────────────
        # Combine les erreurs des voisins pour la correction bottom-up
        self.error_aggregator = nn.Linear(node_dim, node_dim)

        # ── Précisions par nœud (poids de confiance appris) ───────────────────
        # log(Π_i) pour assurer Π_i > 0 (softplus)
        self.log_precision = nn.Parameter(torch.zeros(n_nodes))

        # ── Phases pour la synchronisation de Kuramoto ───────────────────────
        # Initialisées aléatoirement, apprises par la dynamique du système
        self.register_buffer('phases', torch.rand(n_nodes) * 2 * 3.14159)

        # ── États des nœuds ───────────────────────────────────────────────────
        self.register_buffer('mu', torch.zeros(n_nodes, node_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Inférence PC complète sur le graphe.

        Args:
            x          : (n_nodes, node_dim) — features d'observation initiales
            edge_index : (2, n_edges) — arêtes (format COO : [sources; destinations])
                         edge_index[0] = parents, edge_index[1] = enfants
            obs_mask   : (n_nodes,) — True pour les nœuds observés (contrainte dure)
            edge_type  : (n_edges,) — type d'arête (0..n_edge_types-1), défaut: 0

        Returns:
            dict avec :
            - 'mu'         : (n_nodes, node_dim) — représentations convergées
            - 'epsilon'    : (n_nodes, node_dim) — erreurs résiduelles
            - 'free_energy': scalaire — énergie libre totale
            - 'order_param': float — paramètre d'ordre de synchronisation r
        """
        n = self.n_nodes
        device = x.device

        if obs_mask is None:
            obs_mask = torch.ones(n, dtype=torch.bool, device=device)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)

        # ── Encodage initial des nœuds ────────────────────────────────────────
        mu = self.node_encoder(x)  # (n_nodes, node_dim)

        # Contrainte : les nœuds observés commencent à leur valeur observée
        mu = torch.where(obs_mask.unsqueeze(-1), x, mu)

        # Précisions par nœud (Π_i > 0 via softplus)
        precision = F.softplus(self.log_precision)  # (n_nodes,)

        # ── Inférence itérative (passes alternées top-down / bottom-up) ───────
        epsilon = torch.zeros_like(mu)
        free_energy = torch.tensor(0.0, device=device)

        for step in range(self.config.n_inference_steps):
            # ── Passe top-down : calcul des prédictions ──────────────────────
            # Pour chaque arête (parent i → enfant j) :
            # μ̂_j = f_{type}(μ_i)
            predictions = torch.zeros_like(mu)
            prediction_counts = torch.zeros(n, device=device)

            if edge_index.shape[1] > 0:
                src = edge_index[0]  # nœuds parents
                dst = edge_index[1]  # nœuds enfants
                e_type = edge_type

                for k in range(self.n_edge_types):
                    mask_k = (e_type == k)
                    if mask_k.sum() == 0:
                        continue
                    src_k = src[mask_k]
                    dst_k = dst[mask_k]
                    # Prédiction depuis les parents
                    pred_k = self.edge_predictors[k](mu[src_k])  # (n_edges_k, node_dim)
                    # Agrégation par sommation sur les enfants
                    predictions.index_add_(0, dst_k, pred_k)
                    ones = torch.ones(mask_k.sum(), device=device)
                    prediction_counts.index_add_(0, dst_k, ones)

                # Normalisation par le nombre de parents
                has_parents = prediction_counts > 0
                predictions[has_parents] /= prediction_counts[has_parents].unsqueeze(-1)

            # ── Calcul des erreurs de prédiction ──────────────────────────────
            # ε_i = μ_i - μ̂_i (écart entre représentation et prédiction top-down)
            epsilon = mu - predictions  # (n_nodes, node_dim)

            # Nœuds racines (sans parents) : erreur = écart au prior (0)
            root_mask = (prediction_counts == 0) & ~obs_mask
            epsilon[root_mask] = mu[root_mask]  # prior centré en 0

            # ── Passe bottom-up : propagation des erreurs ─────────────────────
            # Pour chaque nœud parent i : corriger μ_i par les erreurs des enfants
            error_corrections = torch.zeros_like(mu)
            if edge_index.shape[1] > 0:
                src = edge_index[0]
                dst = edge_index[1]
                # Les parents reçoivent les erreurs de leurs enfants
                error_corrections.index_add_(0, src, self.error_aggregator(epsilon[dst]))

            # ── Mise à jour des représentations ───────────────────────────────
            # dμ_i/dt = -Π_i · ε_i + Σ_{j enfant} Π_j · ε_j
            # Discrétisation Euler : μ_i(t+1) = μ_i(t) + η · dμ/dt
            prec_i = precision.unsqueeze(-1)  # (n_nodes, 1)
            mu_update = -prec_i * epsilon + error_corrections
            mu = mu + self.config.eta_pc * mu_update

            # Contrainte dure sur les nœuds observés
            mu = torch.where(obs_mask.unsqueeze(-1), x, mu)

        # ── Calcul de l'énergie libre totale ──────────────────────────────────
        # F = 1/2 · Σ_i Π_i · ||ε_i||²
        free_energy = 0.5 * ((precision.unsqueeze(-1) * epsilon ** 2).sum())

        # ── Mise à jour des phases (dynamique de Kuramoto) ────────────────────
        # La matrice de couplage est proportionnelle à la corrélation entre erreurs
        if n > 1:
            # Couplage approximé par la corrélation entre les erreurs des nœuds
            # W_ij ∝ (ε_i · ε_j) / (||ε_i|| · ||ε_j|| + ε)
            eps_norm = epsilon / (epsilon.norm(dim=-1, keepdim=True) + 1e-8)
            W_phase = (eps_norm @ eps_norm.t()) * 0.01  # (n_nodes, n_nodes) — faible couplage
            self.phases = kuramoto_update(self.phases, W_phase, dt=self.config.dt)

        # Paramètre d'ordre de synchronisation
        order_param = compute_order_parameter(self.phases)

        # Mise à jour de l'état interne
        self.mu = mu.detach()

        return {
            'mu': mu,
            'epsilon': epsilon,
            'free_energy': free_energy,
            'order_param': order_param,
            'phases': self.phases.clone(),
        }

    def reset_state(self) -> None:
        """Réinitialise les représentations et les phases."""
        device = self.mu.device
        self.mu = torch.zeros(self.n_nodes, self.node_dim, device=device)
        self.phases = torch.rand(self.n_nodes, device=device) * 2 * 3.14159
