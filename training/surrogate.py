"""
training/surrogate.py — Gradient surrogate pour la rétropropagation à travers les spikes.

Problème fondamental : la fonction de Heaviside H(V - V_threshold) utilisée
pour générer les spikes est non-différentiable (dérivée nulle partout sauf en 0
où elle est infinie). On utilise une approximation lisse (surrogate gradient)
qui préserve la passe avant (forward) binaire tout en fournissant un gradient
utilisable en rétropropagation.

Référence : Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """
    Fonction de spike avec gradient surrogate.

    Passe avant  : H(x) — Heaviside discrète, retourne {0, 1}
    Passe arrière : σ'(x) = 1 / (β · (1 + |x/β|)²)

    Ce surrogate est une approximation de la dérivée d'une sigmoïde étendue.
    Il est non-nul dans une fenêtre autour du seuil, permettant au gradient
    de se propager à travers les couches spikantes.

    Paramètre β contrôle la largeur de la fenêtre :
    - β petit → gradient concentré (approximation précise mais vanishing gradient)
    - β grand → gradient étalé (meilleure propagation mais biais d'approximation)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Args:
            x    : tension membranaire centrée sur le seuil (V - V_threshold)
            beta : largeur de la fenêtre du gradient surrogate

        Returns:
            spike : tenseur binaire {0, 1} — 1 si x > 0 (V dépasse le seuil)
        """
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Gradient surrogate : σ'(x) = 1 / (β · (1 + |x/β|)²)

        Intuition : on approxime la Heaviside par une sigmoïde de pente β⁻¹,
        et on utilise sa dérivée comme gradient de substitution.
        """
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        # Dérivée de la pseudo-sigmoïde au point x
        surrogate_grad = 1.0 / (beta * (1.0 + (x / beta).abs()) ** 2)
        return grad_output * surrogate_grad, None


def heaviside_surrogate(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Interface fonctionnelle pour SurrogateSpike.

    Args:
        x    : (batch, n_neurons) — V - V_threshold, centré sur 0
        beta : paramètre de largeur du surrogate

    Returns:
        spikes : (batch, n_neurons) ∈ {0.0, 1.0} (float pour autograd)

    Exemple:
        >>> V = torch.tensor([-0.5, 0.0, 0.3, 1.2])
        >>> spikes = heaviside_surrogate(V - 0.0)
        >>> # spikes ≈ [0, 0, 1, 1] (discrétisation en forward)
    """
    return SurrogateSpike.apply(x, beta)


class SurrogateHeaviside(nn.Module):
    """
    Module nn.Module encapsulant le surrogate gradient pour utilisation
    dans des architectures séquentielles.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return heaviside_surrogate(x, self.beta)
