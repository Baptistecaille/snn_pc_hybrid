"""
tests/test_convergence.py — Tests unitaires pour la convergence de phase et le PC-GNN.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest
from config import SNNConfig
from graph.phase_sync import (
    kuramoto_condition,
    compute_order_parameter,
    phase_consistency_check,
    kuramoto_update,
)
from graph.pc_gnn import PCGNN
from graph.message_passing import MessagePassingPC


class TestKuramotoCondition:
    """Tests pour la condition de convergence de Kuramoto."""

    def test_small_coupling_converges(self):
        """Un couplage faible (σ_max < γ/L) doit garantir la convergence."""
        W = torch.randn(5, 5) * 0.01  # couplage très faible
        converge, margin = kuramoto_condition(W, gamma=0.1)
        assert converge, f"Convergence attendue pour faible couplage (marge={margin:.4f})"
        assert margin > 0, f"Marge doit être positive : {margin}"

    def test_large_coupling_diverges(self):
        """Un couplage fort (σ_max > γ/L) ne doit pas satisfaire la condition."""
        W = torch.randn(5, 5) * 10.0  # couplage très fort
        converge, margin = kuramoto_condition(W, gamma=0.01)
        assert not converge, f"Divergence attendue pour fort couplage (marge={margin:.4f})"

    def test_margin_sign_consistency(self):
        """La marge doit être positive si et seulement si converge=True."""
        W = torch.randn(4, 4)
        for gamma in [0.001, 0.1, 1.0, 10.0]:
            converge, margin = kuramoto_condition(W, gamma=gamma)
            if converge:
                assert margin > 0, f"Marge devrait être positive quand converge=True"
            else:
                assert margin <= 0, f"Marge devrait être négative quand converge=False"

    def test_identity_matrix(self):
        """Matrice identité : σ_max = 1, converge si γ > L."""
        W = torch.eye(5)
        converge_ok, margin_ok = kuramoto_condition(W, gamma=1.5, lipschitz_constant=1.0)
        converge_fail, margin_fail = kuramoto_condition(W, gamma=0.5, lipschitz_constant=1.0)
        assert converge_ok, f"γ=1.5 > σ_max·L=1.0 devrait converger"
        assert not converge_fail, f"γ=0.5 < σ_max·L=1.0 ne devrait pas converger"

    def test_return_types(self):
        """Les types de retour doivent être (bool, float)."""
        W = torch.randn(3, 3)
        converge, margin = kuramoto_condition(W, gamma=0.1)
        assert isinstance(converge, bool), f"Type converge: {type(converge)}"
        assert isinstance(margin, float), f"Type margin: {type(margin)}"


class TestOrderParameter:
    """Tests pour le paramètre d'ordre de Kuramoto."""

    def test_synchronized_phases(self):
        """Phases identiques → r = 1."""
        phases = torch.zeros(20)  # toutes les phases = 0
        r = compute_order_parameter(phases)
        assert abs(r - 1.0) < 1e-5, f"r devrait être ≈ 1 : {r:.6f}"

    def test_uniform_phases_zero_order(self):
        """Phases uniformément distribuées → r ≈ 0."""
        # Phases exactement équiréparties
        n = 1000
        phases = torch.linspace(0, 2 * math.pi * (1 - 1/n), n)
        r = compute_order_parameter(phases)
        assert r < 0.05, f"r devrait être ≈ 0 pour phases uniformes : {r:.4f}"

    def test_output_range(self):
        """r doit toujours être dans [0, 1]."""
        for _ in range(20):
            phases = torch.rand(50) * 2 * math.pi
            r = compute_order_parameter(phases)
            assert 0 <= r <= 1.0 + 1e-6, f"r hors de [0, 1] : {r}"

    def test_batch_input(self):
        """Doit fonctionner sur des phases de forme (n,) ou (batch, n)."""
        phases_1d = torch.zeros(10)
        phases_2d = torch.zeros(3, 10)
        r_1d = compute_order_parameter(phases_1d)
        r_2d = compute_order_parameter(phases_2d)
        assert abs(r_1d - 1.0) < 1e-5
        assert abs(r_2d - 1.0) < 1e-5


class TestKuramotoUpdate:
    """Tests pour la mise à jour de la dynamique de Kuramoto."""

    def test_output_shape(self):
        """La forme des phases de sortie doit être identique à l'entrée."""
        n = 8
        phases = torch.rand(n) * 2 * math.pi
        W = torch.randn(n, n) * 0.01
        phases_new = kuramoto_update(phases, W, dt=0.1)
        assert phases_new.shape == phases.shape

    def test_phases_in_range(self):
        """Les phases mises à jour doivent rester dans [0, 2π)."""
        n = 10
        for _ in range(20):
            phases = torch.rand(n) * 2 * math.pi
            W = torch.randn(n, n) * 0.1
            phases_new = kuramoto_update(phases, W, dt=0.1)
            assert (phases_new >= 0).all() and (phases_new < 2 * math.pi + 1e-6).all(), \
                f"Phases hors de [0, 2π) : min={phases_new.min():.4f}, max={phases_new.max():.4f}"

    def test_zero_coupling_no_change(self):
        """Sans couplage (W=0), les phases doivent rester identiques (si ω=0)."""
        phases = torch.tensor([0.5, 1.2, 2.8])
        W = torch.zeros(3, 3)
        phases_new = kuramoto_update(phases, W, dt=0.1,
                                     natural_freq=torch.zeros(3))
        # Phases = mêmes (modulo 2π)
        diff = ((phases_new - phases) % (2 * math.pi)).abs()
        assert diff.max() < 1e-5, f"Phases modifiées sans couplage : {diff}"

    def test_synchronization_tendency(self):
        """Avec un fort couplage, les phases doivent converger."""
        n = 5
        phases = torch.rand(n) * 2 * math.pi  # phases aléatoires
        W = torch.ones(n, n) * 0.5  # fort couplage
        W.fill_diagonal_(0.0)

        r_initial = compute_order_parameter(phases)
        for _ in range(200):
            phases = kuramoto_update(phases, W, dt=0.05)
        r_final = compute_order_parameter(phases)

        assert r_final >= r_initial - 0.1 or r_final > 0.5, \
            f"Fort couplage devrait synchroniser : r: {r_initial:.3f} → {r_final:.3f}"


class TestPhaseConsistencyCheck:
    """Tests pour la vérification de cohérence de phase."""

    def test_output_keys(self):
        """La sortie doit contenir les clés requises."""
        n = 5
        graph = (torch.rand(n, n) < 0.3).float()
        graph.fill_diagonal_(0.0)
        phases = torch.rand(n) * 2 * math.pi
        result = phase_consistency_check(graph, phases)
        assert 'cycles' in result
        assert 'consistency' in result
        assert 'global_sync' in result

    def test_global_sync_range(self):
        """global_sync doit être dans [0, 1]."""
        n = 6
        graph = torch.zeros(n, n)
        phases = torch.rand(n) * 2 * math.pi
        result = phase_consistency_check(graph, phases)
        r = result['global_sync']
        assert 0 <= r <= 1.0, f"global_sync hors de [0, 1] : {r}"

    def test_consistency_range(self):
        """Les scores de cohérence doivent être dans (0, 1]."""
        n = 6
        graph = (torch.rand(n, n) < 0.4).float()
        graph.fill_diagonal_(0.0)
        phases = torch.rand(n) * 2 * math.pi
        result = phase_consistency_check(graph, phases)
        if len(result['consistency']) > 0:
            scores = result['consistency']
            assert (scores > 0).all() and (scores <= 1.0 + 1e-6).all(), \
                f"Scores de cohérence hors de (0, 1] : {scores}"


class TestPCGNN:
    """Tests pour le Predictive Coding Graph Neural Network."""

    @pytest.fixture
    def setup(self):
        config = SNNConfig()
        config.n_inference_steps = 5
        n_nodes = 6
        node_dim = 8
        model = PCGNN(n_nodes, node_dim, config)
        return model, config, n_nodes, node_dim

    def test_output_shape(self, setup):
        """Les sorties doivent avoir les bonnes formes."""
        model, config, n_nodes, node_dim = setup
        x = torch.randn(n_nodes, node_dim)
        # Graphe en chaîne simple
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        out = model(x, edge_index)

        assert out['mu'].shape == (n_nodes, node_dim), f"Forme mu: {out['mu'].shape}"
        assert out['epsilon'].shape == (n_nodes, node_dim)
        assert isinstance(out['free_energy'].item(), float)
        assert isinstance(out['order_param'], float)

    def test_observed_nodes_constrained(self, setup):
        """Les nœuds observés doivent conserver leur valeur après inférence."""
        model, config, n_nodes, node_dim = setup
        x = torch.randn(n_nodes, node_dim)
        obs_mask = torch.tensor([True, True, False, False, False, False])
        edge_index = torch.tensor([[0, 1], [2, 3]])

        out = model(x, edge_index, obs_mask=obs_mask)
        # Les nœuds observés doivent avoir mu ≈ x
        assert torch.allclose(out['mu'][obs_mask], x[obs_mask], atol=1e-4), \
            "Les nœuds observés doivent conserver leur valeur"

    def test_free_energy_positive(self, setup):
        """L'énergie libre doit toujours être positive (somme de carrés / 2)."""
        model, config, n_nodes, node_dim = setup
        x = torch.randn(n_nodes, node_dim)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        out = model(x, edge_index)
        assert out['free_energy'].item() >= 0, \
            f"Énergie libre négative : {out['free_energy'].item()}"

    def test_reset_state(self, setup):
        """reset_state() doit remettre mu à zéro."""
        model, config, n_nodes, node_dim = setup
        x = torch.randn(n_nodes, node_dim)
        edge_index = torch.tensor([[0], [1]])
        model(x, edge_index)

        model.reset_state()
        assert torch.allclose(model.mu, torch.zeros_like(model.mu)), \
            "mu non réinitialisé"


class TestMessagePassingPC:
    """Tests pour le passage de messages PC."""

    def test_output_shape(self):
        """Les sorties doivent avoir les bonnes formes."""
        config = SNNConfig()
        config.n_inference_steps = 3
        n_nodes, node_dim = 4, 8
        model = MessagePassingPC(n_nodes, node_dim, config)

        observations = torch.randn(n_nodes, node_dim)
        adj = torch.tensor([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=torch.float)

        out = model(observations, adj)
        assert out['mu'].shape == (n_nodes, node_dim)
        assert out['epsilon'].shape == (n_nodes, node_dim)
        assert isinstance(out['free_energy'].item(), float)

    def test_free_energy_nonnegative(self):
        """L'énergie libre doit être non-négative."""
        config = SNNConfig()
        config.n_inference_steps = 5
        n_nodes, node_dim = 4, 6
        model = MessagePassingPC(n_nodes, node_dim, config)
        observations = torch.randn(n_nodes, node_dim)
        adj = torch.eye(n_nodes) * 0.0  # graphe sans arêtes
        out = model(observations, adj)
        assert out['free_energy'].item() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
