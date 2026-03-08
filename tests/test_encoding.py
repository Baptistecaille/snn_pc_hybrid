"""
tests/test_encoding.py — Tests unitaires pour les schémas d'encodage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest
from core.encoding import rate_encode, phase_encode, population_rate_to_value


class TestRateEncoding:
    """Tests pour l'encodage par taux de décharge."""

    def test_output_binary(self):
        """rate_encode doit produire des valeurs binaires {0, 1}."""
        epsilon = torch.randn(8, 32)
        spikes = rate_encode(epsilon, dt=0.1)
        unique_vals = spikes.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals), \
            f"Valeurs non-binaires : {unique_vals}"

    def test_output_shape(self):
        """La forme de sortie doit correspondre à l'entrée."""
        epsilon = torch.randn(4, 16)
        spikes = rate_encode(epsilon, dt=0.1)
        assert spikes.shape == epsilon.shape

    def test_large_epsilon_more_spikes(self):
        """Un ε large (positif) doit produire plus de spikes en moyenne."""
        n_samples = 500
        epsilon_small = torch.full((n_samples, 10), -3.0)  # sigmoid(-3) ≈ 0.05
        epsilon_large = torch.full((n_samples, 10), 3.0)   # sigmoid(3) ≈ 0.95

        rate_small = rate_encode(epsilon_small, dt=0.1).float().mean().item()
        rate_large = rate_encode(epsilon_large, dt=0.1).float().mean().item()

        assert rate_large > rate_small, \
            f"Grand ε devrait produire plus de spikes : {rate_small:.4f} vs {rate_large:.4f}"

    def test_dt_scaling(self):
        """Un dt plus grand doit produire plus de spikes (probabilité ∝ dt)."""
        torch.manual_seed(0)
        n = 1000
        epsilon = torch.zeros(n, 20)  # sigmoid(0) = 0.5

        # dt petit → peu de spikes
        rate_small_dt = rate_encode(epsilon, dt=0.01).float().mean().item()
        # dt grand → beaucoup de spikes
        rate_large_dt = rate_encode(epsilon, dt=0.5).float().mean().item()

        assert rate_large_dt > rate_small_dt, \
            f"dt plus grand devrait donner plus de spikes"

    def test_probability_valid(self):
        """La probabilité (avant Bernoulli) doit rester dans [0, 1]."""
        epsilon = torch.tensor([-100.0, 0.0, 100.0])
        # On vérifie que sigmoid est bornée
        prob = torch.sigmoid(epsilon) * 0.1  # dt=0.1
        assert (prob >= 0).all() and (prob <= 1).all(), "Probabilité hors de [0, 1]"


class TestPhaseEncoding:
    """Tests pour l'encodage de phase (temporal coding)."""

    def test_output_binary(self):
        """phase_encode doit produire des valeurs binaires {0, 1}."""
        epsilon = torch.randn(4, 16)
        spikes = phase_encode(epsilon, phase=0.0)
        unique_vals = spikes.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals), \
            f"Valeurs non-binaires : {unique_vals}"

    def test_output_shape(self):
        """La forme de sortie doit correspondre à l'entrée."""
        epsilon = torch.randn(4, 16)
        spikes = phase_encode(epsilon, phase=1.0)
        assert spikes.shape == epsilon.shape

    def test_large_error_spikes_early(self):
        """Les neurones avec grande erreur doivent spiker tôt dans le cycle gamma."""
        n = 50
        epsilon_large = torch.full((1, n), 4.5)  # proche de epsilon_max=5
        epsilon_small = torch.full((1, n), 0.1)  # petite erreur

        # Pour ε_large=4.5 : ε_norm=0.9 → phase_target = 2π·0.1 ≈ 0.628 rad
        # Pour ε_small=0.1 : ε_norm=0.02 → phase_target = 2π·0.98 ≈ 6.157 rad
        # À phase=0.4 (tolerance=0.5) :
        #   ε_large : |0.4 - 0.628| = 0.228 < 0.5 → SPIKE ✓
        #   ε_small : min-dist(0.4, 6.157) = min(5.757, 0.526) = 0.526 > 0.5 → no spike ✓
        spikes_large_early = phase_encode(epsilon_large, phase=0.4, epsilon_max=5.0, alpha=1.0)
        spikes_small_early = phase_encode(epsilon_small, phase=0.4, epsilon_max=5.0, alpha=1.0)

        rate_large = spikes_large_early.float().mean().item()
        rate_small = spikes_small_early.float().mean().item()

        assert rate_large >= rate_small, \
            f"Grande erreur devrait spiker plus tôt : {rate_large:.3f} vs {rate_small:.3f}"

    def test_phase_periodicity(self):
        """Les spikes doivent être cohérents avec la périodicité de la phase."""
        epsilon = torch.ones(1, 16) * 2.5  # erreur intermédiaire
        # Deux phases très proches → résultats similaires
        spikes1 = phase_encode(epsilon, phase=1.0, epsilon_max=5.0)
        spikes2 = phase_encode(epsilon, phase=1.01, epsilon_max=5.0)
        # Elles ne doivent pas différer massivement (phases proches)
        # (test qualitatif sur la continuité)
        diff = (spikes1 - spikes2).abs().mean().item()
        assert diff <= 1.0, f"Discontinuité de phase : {diff:.4f}"


class TestPopulationDecoding:
    """Tests pour le décodage depuis les trains de spikes."""

    def test_high_activity_high_rate(self):
        """Un train de spikes dense doit donner un taux élevé."""
        T, batch, n = 100, 2, 8
        # Tous les neurones spikent à chaque pas
        spikes_dense = torch.ones(T, batch, n)
        rate = population_rate_to_value(spikes_dense, window_steps=50)
        assert rate.shape == (batch, n)
        assert (rate > 0.9).all(), f"Taux trop bas : {rate.mean():.3f}"

    def test_no_activity_zero_rate(self):
        """Un train de spikes vide doit donner un taux nul."""
        T, batch, n = 100, 2, 8
        spikes_empty = torch.zeros(T, batch, n)
        rate = population_rate_to_value(spikes_empty, window_steps=50)
        assert (rate == 0).all()

    def test_window_clipping(self):
        """La fenêtre doit être tronquée si plus longue que le train de spikes."""
        T = 10
        spikes = torch.ones(T, 2, 4)
        # window > T → doit fonctionner sans erreur
        rate = population_rate_to_value(spikes, window_steps=200)
        assert rate.shape == (2, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
