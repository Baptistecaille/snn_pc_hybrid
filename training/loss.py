"""
training/loss.py — Énergie libre variationnelle et fonctions de perte PC.

L'énergie libre variationnelle F est la fonction objectif centrale du
Predictive Coding (Friston, 2005). Elle quantifie l'écart entre les
prédictions du modèle et les observations, pondéré par la précision
(inverse de la variance) de chaque couche.

F = 1/2 · Σ_ℓ Σ_i (ε_i^(ℓ))² / σ_ℓ²

Minimiser F revient à :
1. Minimiser les erreurs de prédiction (inférence)
2. Optimiser les poids pour que les prédictions soient plus précises (apprentissage)
"""

import torch
import torch.nn as nn
from typing import Optional


def variational_free_energy(
    epsilon_wernicke: torch.Tensor,   # (batch, dim_W)
    epsilon_broca: torch.Tensor,      # (batch, dim_B)
    sigma_W: float = 1.0,
    sigma_B: float = 1.0,
    epsilon_arcuate: Optional[torch.Tensor] = None,
    sigma_A: float = 1.0,
) -> torch.Tensor:
    """
    Énergie libre variationnelle totale du système Wernicke-Broca.

    F = (1/2σ_W²) · ||ε_W||² + (1/2σ_B²) · ||ε_B||²

    Les σ² agissent comme des poids de précision : une faible variance
    impose une forte contrainte sur les erreurs de cette couche.

    Args:
        epsilon_wernicke : erreurs de prédiction du module Wernicke (batch, dim_W)
        epsilon_broca    : erreurs de prédiction du module Broca (batch, dim_B)
        sigma_W          : écart-type du prior de Wernicke
        sigma_B          : écart-type du prior de Broca
        epsilon_arcuate  : erreurs du fascicule arqué (optionnel) (batch, dim_A)
        sigma_A          : écart-type du prior du fascicule arqué

    Returns:
        F : énergie libre scalaire, moyennée sur le batch
    """
    F_W = 0.5 * (epsilon_wernicke ** 2).sum(dim=-1) / (sigma_W ** 2)
    F_B = 0.5 * (epsilon_broca ** 2).sum(dim=-1) / (sigma_B ** 2)

    F_total = F_W + F_B

    if epsilon_arcuate is not None:
        F_A = 0.5 * (epsilon_arcuate ** 2).sum(dim=-1) / (sigma_A ** 2)
        F_total = F_total + F_A

    return F_total.mean()


def prediction_error_loss(
    prediction: torch.Tensor,    # (batch, dim)
    target: torch.Tensor,        # (batch, dim)
    precision: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Erreur de prédiction pondérée par la précision.

    L_PC = (precision / 2) · ||prediction - target||²

    C'est la version per-sample de l'énergie libre pour une seule couche.

    Args:
        prediction : prédiction du modèle μ (batch, dim)
        target     : valeur observée x (batch, dim)
        precision  : précision = 1/σ² (inverse de la variance du prior)
        reduction  : 'mean' | 'sum' | 'none'

    Returns:
        loss : scalaire (ou tenseur si reduction='none')
    """
    squared_error = (prediction - target) ** 2
    loss = (precision / 2.0) * squared_error.sum(dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def spike_regularization(
    spikes: torch.Tensor,
    target_rate: float = 0.1,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Régularisation du taux de décharge moyen.

    Pénalise les réseaux qui spikent trop peu (silence pathologique)
    ou trop (épilepsie artificielle). Cible un taux moyen de 10%.

    L_reg = weight · (mean_rate - target_rate)²

    Args:
        spikes      : (batch, n_neurons) ou (batch, T, n_neurons) — spikes binaires
        target_rate : taux de décharge cible (fraction de 1)
        weight      : poids de la régularisation

    Returns:
        loss : scalaire
    """
    mean_rate = spikes.float().mean()
    return weight * (mean_rate - target_rate) ** 2


def phase_loss(
    epsilon_W: torch.Tensor,
    epsilon_B: torch.Tensor,
    r_W: float,
    r_B: float,
    phase: str,
    sigma_W: float = 1.0,
    sigma_B: float = 1.0,
    lambda_sync: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Loss adaptée à chaque phase du curriculum d'entraînement.

    Stratégie par phase :
    - 'bootstrap'     : F_W uniquement (Broca gelé, Wernicke seul apprend)
    - 'wikipedia_*'   : F_W + F_B + pénalité de désynchronisation
    - 'oscar_*'       : F_W + F_B + cohérence de phase W↔B (même formule
                        que wikipedia mais avec des séquences plus longues
                        et un cosine annealing)

    La pénalité de synchronisation vaut :
        L_sync = λ_sync · (2 - r_W - r_B)
    Elle est nulle quand r_W = r_B = 1 (synchronisation totale) et maximale
    quand r_W = r_B = 0 (désynchronisation complète). Ce terme encourage les
    deux modules à maintenir une cohérence de phase pendant l'inférence.

    Args:
        epsilon_W    : (batch, dim_W) — erreurs de Wernicke
        epsilon_B    : (batch, dim_B) — erreurs de Broca
        r_W          : paramètre d'ordre Kuramoto de Wernicke ∈ [0, 1]
        r_B          : paramètre d'ordre Kuramoto de Broca ∈ [0, 1]
        phase        : identifiant de la phase courante du curriculum
        sigma_W      : précision du prior Wernicke
        sigma_B      : précision du prior Broca
        lambda_sync  : poids du terme de synchronisation (défaut 0.1)

    Returns:
        loss      : tenseur scalaire différentiable
        breakdown : dict {'F_W', 'F_B', 'sync'} pour le logging CSV
    """
    F_W_term = 0.5 * (epsilon_W ** 2).sum(dim=-1).mean() / (sigma_W ** 2)
    F_B_term = 0.5 * (epsilon_B ** 2).sum(dim=-1).mean() / (sigma_B ** 2)
    sync_penalty = lambda_sync * (2.0 - r_W - r_B)   # scalaire Python

    if phase == 'bootstrap':
        # Broca gelé : on n'optimise que l'erreur de Wernicke
        loss = F_W_term
        breakdown = {
            'F_W':  float(F_W_term.item()),
            'F_B':  0.0,
            'sync': 0.0,
        }

    elif phase.startswith('wikipedia'):
        # Les deux modules + pénalité de désynchronisation
        loss = F_W_term + F_B_term + sync_penalty
        breakdown = {
            'F_W':  float(F_W_term.item()),
            'F_B':  float(F_B_term.item()),
            'sync': float(sync_penalty),
        }

    elif phase.startswith('oscar'):
        # Identique à wikipedia mais avec cosine annealing externe (géré par le trainer)
        loss = F_W_term + F_B_term + sync_penalty
        breakdown = {
            'F_W':  float(F_W_term.item()),
            'F_B':  float(F_B_term.item()),
            'sync': float(sync_penalty),
        }

    else:
        # Phase inconnue : énergie libre totale par défaut
        loss = variational_free_energy(epsilon_W, epsilon_B, sigma_W, sigma_B)
        breakdown = {
            'F_W':  float(F_W_term.item()),
            'F_B':  float(F_B_term.item()),
            'sync': 0.0,
        }

    return loss, breakdown


class FreeEnergyLoss(nn.Module):
    """
    Module nn.Module pour l'énergie libre variationnelle totale.
    Agrège les contributions de toutes les couches avec leurs précisions respectives.
    """

    def __init__(self, sigma_W: float = 1.0, sigma_B: float = 1.0, sigma_A: float = 1.0):
        super().__init__()
        self.sigma_W = sigma_W
        self.sigma_B = sigma_B
        self.sigma_A = sigma_A

    def forward(
        self,
        epsilon_wernicke: torch.Tensor,
        epsilon_broca: torch.Tensor,
        epsilon_arcuate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return variational_free_energy(
            epsilon_wernicke=epsilon_wernicke,
            epsilon_broca=epsilon_broca,
            sigma_W=self.sigma_W,
            sigma_B=self.sigma_B,
            epsilon_arcuate=epsilon_arcuate,
            sigma_A=self.sigma_A,
        )
