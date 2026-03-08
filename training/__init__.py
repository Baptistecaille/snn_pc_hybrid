"""
training/ — Fonctions de perte et gradients surrogate pour l'apprentissage SNN-PC.
"""

from training.loss import variational_free_energy, prediction_error_loss
from training.surrogate import SurrogateSpike, heaviside_surrogate

__all__ = [
    "variational_free_energy",
    "prediction_error_loss",
    "SurrogateSpike",
    "heaviside_surrogate",
]
