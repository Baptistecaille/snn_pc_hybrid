"""
tests/test_neuron.py — Tests unitaires pour LIFNeuron et le surrogate gradient.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from config import SNNConfig
from core.neuron import LIFNeuron
from training.surrogate import heaviside_surrogate, SurrogateSpike


class TestSurrogateGradient:
    """Tests pour le surrogate gradient de Heaviside."""

    def test_forward_binary(self):
        """La passe avant doit produire des valeurs binaires {0.0, 1.0}."""
        x = torch.tensor([-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0])
        spikes = heaviside_surrogate(x)
        unique_vals = spikes.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals), \
            f"Valeurs non-binaires : {unique_vals}"

    def test_forward_threshold(self):
        """Les valeurs > 0 doivent donner 1, les valeurs <= 0 doivent donner 0."""
        x = torch.tensor([-1.0, -0.001, 0.001, 1.0])
        spikes = heaviside_surrogate(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(spikes, expected), f"Spikes: {spikes}, attendu: {expected}"

    def test_backward_nonzero(self):
        """Le gradient surrogate doit être non-nul dans une fenêtre autour de 0."""
        x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        spikes = heaviside_surrogate(x, beta=1.0)
        loss = spikes.sum()
        loss.backward()
        # Le gradient doit être non-nul (surrogate actif)
        assert x.grad is not None, "Gradient None"
        assert (x.grad != 0).any(), "Gradient nul pour toutes les valeurs"

    def test_backward_vanishes_far(self):
        """Le gradient doit être très petit loin du seuil."""
        x_far = torch.tensor([-10.0, 10.0], requires_grad=True)
        spikes = heaviside_surrogate(x_far, beta=1.0)
        spikes.sum().backward()
        assert x_far.grad.abs().max() < 0.01, \
            f"Gradient trop grand loin du seuil : {x_far.grad}"

    def test_beta_effect(self):
        """Un β plus grand doit donner un gradient plus étalé."""
        x = torch.tensor([1.0], requires_grad=True)

        # Beta petit → gradient concentré
        spikes_small = heaviside_surrogate(x, beta=0.1)
        spikes_small.backward()
        grad_small = x.grad.clone()

        x.grad = None
        # Beta grand → gradient étalé (valeur plus grande à x=1)
        spikes_large = heaviside_surrogate(x, beta=2.0)
        spikes_large.backward()
        grad_large = x.grad.clone()

        # À x=1 > β_small, le gradient est plus grand pour β grand
        assert grad_large.abs() > grad_small.abs(), \
            "β grand devrait donner un gradient plus étalé"


class TestLIFNeuron:
    """Tests pour le neurone LIF augmenté PC."""

    @pytest.fixture
    def setup(self):
        config = SNNConfig()
        config.dt = 0.1
        n_neurons = 32
        neuron = LIFNeuron(n_neurons, config)
        return neuron, config, n_neurons

    def test_output_shape(self, setup):
        """forward() doit retourner des tenseurs de la bonne forme."""
        neuron, config, n_neurons = setup
        batch = 4
        I_syn = torch.zeros(batch, n_neurons)
        epsilon = torch.zeros(batch, n_neurons)
        spikes, V = neuron(I_syn, epsilon, phase=0.0)
        assert spikes.shape == (batch, n_neurons), f"Forme spikes: {spikes.shape}"
        assert V.shape == (batch, n_neurons), f"Forme V: {V.shape}"

    def test_spikes_binary(self, setup):
        """Les spikes doivent être dans {0, 1} (modulo les valeurs du surrogate en forward)."""
        neuron, config, n_neurons = setup
        batch = 8
        I_syn = torch.randn(batch, n_neurons) * 10  # fort courant pour provoquer des spikes
        epsilon = torch.randn(batch, n_neurons)
        spikes, V = neuron(I_syn, epsilon, phase=0.0)
        unique_vals = spikes.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals), \
            f"Valeurs non-binaires : {unique_vals}"

    def test_membrane_reset_after_spike(self, setup):
        """Le potentiel doit être réinitialisé après un spike."""
        neuron, config, n_neurons = setup
        # Forcer un spike en appliquant un courant très fort
        batch = 1
        I_very_strong = torch.full((batch, n_neurons), 1000.0)
        epsilon = torch.zeros(batch, n_neurons)
        spikes, V = neuron(I_very_strong, epsilon, phase=0.0)

        # Si des spikes ont été émis, leur potentiel doit être v_reset
        spiked = spikes.bool()
        if spiked.any():
            V_spiked = V[spiked]
            assert torch.allclose(V_spiked, torch.full_like(V_spiked, config.v_reset), atol=0.1), \
                f"Potentiel post-spike : {V_spiked.mean():.2f} (attendu {config.v_reset})"

    def test_pc_error_increases_firing(self, setup):
        """Un ε positif (erreur de prédiction) doit augmenter le taux de décharge."""
        neuron, config, n_neurons = setup
        batch = 16
        I_syn = torch.zeros(batch, n_neurons)

        # Test sans erreur PC
        neuron.reset_state()
        epsilon_zero = torch.zeros(batch, n_neurons)
        spikes_zero_list = []
        for _ in range(100):
            s, _ = neuron(I_syn, epsilon_zero, phase=0.0)
            spikes_zero_list.append(s)
        rate_zero = torch.stack(spikes_zero_list).float().mean().item()

        # Test avec erreur PC positive (dépolarisation supplémentaire)
        neuron.reset_state()
        epsilon_pos = torch.ones(batch, n_neurons) * 5.0  # forte erreur positive
        spikes_pos_list = []
        for _ in range(100):
            s, _ = neuron(I_syn, epsilon_pos, phase=0.0)
            spikes_pos_list.append(s)
        rate_pos = torch.stack(spikes_pos_list).float().mean().item()

        assert rate_pos >= rate_zero, \
            f"ε positif devrait augmenter le taux : {rate_zero:.4f} → {rate_pos:.4f}"

    def test_reset_state(self, setup):
        """reset_state() doit remettre V à v_rest et vider l'historique."""
        neuron, config, n_neurons = setup
        batch = 4
        I_syn = torch.randn(batch, n_neurons) * 5.0
        epsilon = torch.randn(batch, n_neurons)

        # Quelques pas de simulation
        for _ in range(10):
            neuron(I_syn, epsilon, phase=0.0)

        neuron.reset_state()

        assert torch.allclose(neuron.V, torch.full_like(neuron.V, config.v_rest)), \
            f"V non réinitialisé : {neuron.V.mean():.2f}"
        assert len(neuron.spike_history) == 0, "Historique non vidé"
        assert neuron.current_time == 0.0, f"Temps non réinitialisé : {neuron.current_time}"

    def test_state_is_tracked_for_full_batch(self, setup):
        """L'état interne doit conserver tout le batch, pas seulement le premier élément."""
        neuron, config, n_neurons = setup
        batch = 3
        I_syn = torch.randn(batch, n_neurons)
        epsilon = torch.randn(batch, n_neurons)

        spikes, V = neuron(I_syn, epsilon, phase=0.0)

        assert neuron.V.shape == (batch, n_neurons)
        assert neuron.I_syn_state.shape == (batch, n_neurons)
        assert neuron.t_last_spike.shape == (batch, n_neurons)
        assert torch.allclose(neuron.V, V.detach())

    def test_reset_state_respects_batch_size(self, setup):
        """reset_state(batch_size=N) doit redimensionner les buffers internes."""
        neuron, config, n_neurons = setup
        neuron.reset_state(batch_size=5)

        assert neuron.V.shape == (5, n_neurons)
        assert neuron.I_syn_state.shape == (5, n_neurons)
        assert neuron.t_last_spike.shape == (5, n_neurons)

    def test_gradient_flows(self, setup):
        """Les gradients doivent se propager à travers les spikes (surrogate)."""
        neuron, config, n_neurons = setup
        batch = 2
        I_syn = torch.randn(batch, n_neurons, requires_grad=True)
        epsilon = torch.zeros(batch, n_neurons)

        spikes, V = neuron(I_syn, epsilon, phase=0.0)
        loss = spikes.sum() + V.sum()
        loss.backward()

        assert I_syn.grad is not None, "Gradient nul pour I_syn"

    def test_firing_rate_computation(self, setup):
        """get_firing_rate() doit retourner un taux ∈ [0, ∞) de la bonne forme."""
        neuron, config, n_neurons = setup
        batch = 4
        I_syn = torch.randn(batch, n_neurons) * 3.0
        epsilon = torch.zeros(batch, n_neurons)

        for _ in range(50):
            neuron(I_syn, epsilon, phase=0.0)

        rates = neuron.get_firing_rate(window_ms=50.0)
        assert rates.shape == (n_neurons,), f"Forme rates: {rates.shape}"
        assert (rates >= 0).all(), "Taux négatif"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
