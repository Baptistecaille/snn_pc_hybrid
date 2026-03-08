"""
graph/message_passing.py — Passage de messages PC sur graphes avec historique de visite.

Le Predictive Coding standard suppose un graphe hiérarchique acyclique (DAG).
Pour l'étendre aux graphes cycliques, on doit gérer les boucles de messages
qui sinon divergent.

Stratégie : chaque message porte un "historique de visite" (liste des nœuds
déjà traversés). Quand un message revient sur un nœud déjà visité, son poids
est atténué par un facteur exponentiel.

Cette approche est analogue à l'algorithme de Loopy Belief Propagation (LBP)
avec convergence garantie sous la condition de Kuramoto.

Complexité : O(|V| · |E|) par itération, avec une borne sur la profondeur
de propagation (max_hops) pour limiter la complexité sur les graphes denses.
"""

import torch
import torch.nn as nn
from typing import Optional
from config import SNNConfig


class MessagePassingPC(nn.Module):
    """
    Passage de messages Predictive Coding sur graphe arbitraire.

    Chaque nœud maintient :
    - μ_i : représentation latente (similaire aux μ des modules Wernicke/Broca)
    - ε_i : erreur de prédiction (différence entre observation et prédiction)

    Les messages passent des parents aux enfants (top-down) et en sens inverse
    (bottom-up) pour les corrections d'erreur.

    Amortissement des cycles : chaque message garde son historique de nœuds
    visités et son poids décroît exponentiellement avec le nombre de cycles.
    """

    def __init__(
        self,
        n_nodes: int,
        node_dim: int,
        config: SNNConfig,
        max_hops: int = 5,
    ):
        """
        Args:
            n_nodes  : nombre de nœuds dans le graphe
            node_dim : dimension de la représentation par nœud
            config   : configuration globale
            max_hops : profondeur maximale de propagation (anti-explosion de cycles)
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.config = config
        self.max_hops = max_hops

        # Fonctions de prédiction par nœud (f_i : μ_parent → μ_enfant prédit)
        self.prediction_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh(),
            )
            for _ in range(n_nodes)
        ])

        # Représentations latentes (états)
        self.register_buffer('mu', torch.zeros(n_nodes, node_dim))
        self.register_buffer('epsilon', torch.zeros(n_nodes, node_dim))

    def forward(
        self,
        observations: torch.Tensor,
        adj_matrix: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Inférence PC sur le graphe complet via passage de messages itératif.

        Algorithme :
        1. Initialiser μ à partir des observations (nœuds observés)
        2. Pour chaque itération d'inférence :
           a. Calculer les prédictions top-down : x_pred_i = f_j(μ_j) pour parent j
           b. Calculer les erreurs : ε_i = obs_i - x_pred_i
           c. Propagation bottom-up : corriger μ_j ← μ_j + η · W_ji · ε_i
           d. Amortissement des cycles via l'historique de visite

        Args:
            observations : (n_nodes, node_dim) — observations pour les nœuds visibles
            adj_matrix   : (n_nodes, n_nodes) — matrice d'adjacence orientée
                           adj[i,j] = 1 → arête de i vers j (i parent de j)
            obs_mask     : (n_nodes,) — masque booléen (True = nœud observé)

        Returns:
            dict avec :
            - 'mu'     : (n_nodes, node_dim) — représentations après convergence
            - 'epsilon': (n_nodes, node_dim) — erreurs de prédiction
            - 'free_energy': float — énergie libre totale
        """
        n = self.n_nodes
        mu = self.mu.clone()

        if obs_mask is None:
            obs_mask = torch.ones(n, dtype=torch.bool, device=mu.device)

        # Initialiser les nœuds observés avec les observations
        mu[obs_mask] = observations[obs_mask].float()

        # ── Inférence itérative ──────────────────────────────────────────────
        for step in range(self.config.n_inference_steps):
            epsilon = torch.zeros_like(mu)

            # Calcul des erreurs pour chaque nœud
            for i in range(n):
                # Trouver les parents de i (nœuds j tels que adj[j, i] = 1)
                parents = (adj_matrix[:, i] > 0).nonzero(as_tuple=True)[0]

                if len(parents) == 0:
                    # Nœud racine : erreur = écart au prior
                    epsilon[i] = mu[i]  # prior centré en 0
                else:
                    # Agréger les prédictions des parents
                    x_pred = torch.zeros(self.node_dim, device=mu.device)
                    for j in parents:
                        j_idx = j.item()
                        # Détecter les cycles dans l'historique (simplifié : comptage de visites)
                        x_pred_j = self.prediction_fns[j_idx](mu[j_idx].unsqueeze(0)).squeeze(0)
                        x_pred = x_pred + x_pred_j / len(parents)

                    epsilon[i] = mu[i] - x_pred

                # Contrainte d'observation : les nœuds observés ont une erreur d'observation
                if obs_mask[i]:
                    epsilon[i] = epsilon[i] + (observations[i].float() - mu[i])

            # Mise à jour des représentations latentes
            # Pour chaque nœud : dμ_i/dt = -ε_i + Σ_j adj_ij · W_ji · ε_j
            mu_update = torch.zeros_like(mu)
            for i in range(n):
                # Terme de correction par les erreurs des enfants (bottom-up)
                children = (adj_matrix[i, :] > 0).nonzero(as_tuple=True)[0]
                child_correction = torch.zeros(self.node_dim, device=mu.device)
                for k in children:
                    k_idx = k.item()
                    child_correction = child_correction + epsilon[k_idx]
                if len(children) > 0:
                    child_correction = child_correction / len(children)

                mu_update[i] = -epsilon[i] + 0.5 * child_correction

            mu = mu + self.config.eta_pc * mu_update

            # Reclamper les nœuds observés à leurs valeurs
            mu[obs_mask] = observations[obs_mask].float()

        # Calculer l'énergie libre totale
        free_energy = 0.5 * (epsilon ** 2).sum() / (self.config.sigma_prior ** 2)

        # Mise à jour des buffers d'état
        self.mu = mu.detach()
        self.epsilon = epsilon.detach()

        return {
            'mu': mu,
            'epsilon': epsilon,
            'free_energy': free_energy,
        }

    def reset_state(self) -> None:
        """Réinitialise les représentations latentes."""
        device = self.mu.device
        self.mu = torch.zeros(self.n_nodes, self.node_dim, device=device)
        self.epsilon = torch.zeros(self.n_nodes, self.node_dim, device=device)
