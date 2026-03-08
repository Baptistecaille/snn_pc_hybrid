"""
experiments/convergence_test.py — Test de convergence sur graphes cycliques.

Objectif : montrer que l'amortissement cyclique du fascicule arqué préserve
la convergence là où le PC standard diverge sur des graphes denses.

Protocole expérimental :
1. Générer des graphes aléatoires avec différentes densités de cycles :
   - Graphe acyclique (DAG) → baseline
   - Graphe avec cycles simples (longueur 3–5)
   - Graphe dense aléatoire (Erdős–Rényi, p=0.3, 0.5, 0.8)

2. Pour chaque topologie :
   a. Simuler le PC-GNN avec amortissement (notre méthode)
   b. Simuler le PC-GNN sans amortissement (baseline)
   c. Mesurer r(t) et F(t) au fil des pas

3. Vérifier la condition de Kuramoto : γ > σ_max(W) · L

Résultat attendu : l'amortissement préserve la convergence là où PC nu diverge.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

from config import SNNConfig
from graph.pc_gnn import PCGNN
from graph.phase_sync import (
    kuramoto_condition,
    phase_consistency_check,
    compute_order_parameter,
    kuramoto_update,
)


# ── Générateurs de graphes ────────────────────────────────────────────────────

def make_dag(n_nodes: int, branching: float = 0.3) -> torch.Tensor:
    """
    Génère un DAG (graphe orienté acyclique) aléatoire.
    Les arêtes vont toujours de l'indice inférieur vers l'indice supérieur.

    Args:
        n_nodes   : nombre de nœuds
        branching : probabilité d'arête entre nœuds i < j

    Returns:
        adj : (n_nodes, n_nodes) — matrice d'adjacence
    """
    adj = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < branching:
                adj[i, j] = 1.0
    return adj


def make_cyclic_graph(n_nodes: int, n_extra_edges: int = 3) -> torch.Tensor:
    """
    Crée un graphe en chaîne auquel on ajoute n_extra_edges arêtes de retour,
    créant des cycles de longueurs variables.

    Args:
        n_nodes       : nombre de nœuds
        n_extra_edges : nombre d'arêtes supplémentaires créant des cycles

    Returns:
        adj : (n_nodes, n_nodes) — matrice d'adjacence
    """
    # Chaîne de base
    adj = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = 1.0

    # Arêtes de retour (cycles)
    added = 0
    attempts = 0
    while added < n_extra_edges and attempts < 100:
        attempts += 1
        j = np.random.randint(0, n_nodes)
        i = np.random.randint(0, n_nodes)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1.0
            added += 1

    return adj


def make_erdos_renyi(n_nodes: int, p: float) -> torch.Tensor:
    """
    Génère un graphe d'Erdős–Rényi orienté G(n, p).

    Args:
        n_nodes : nombre de nœuds
        p       : probabilité de chaque arête

    Returns:
        adj : (n_nodes, n_nodes) — matrice d'adjacence (sans auto-boucles)
    """
    adj = (torch.rand(n_nodes, n_nodes) < p).float()
    adj.fill_diagonal_(0.0)  # pas d'auto-boucles
    return adj


# ── Simulation de convergence ─────────────────────────────────────────────────

def simulate_convergence(
    adj: torch.Tensor,
    config: SNNConfig,
    n_steps: int = 100,
    use_damping: bool = True,
    node_dim: int = 16,
) -> dict[str, list[float]]:
    """
    Simule la dynamique de convergence du PC-GNN sur un graphe donné.

    Args:
        adj        : (n_nodes, n_nodes) — matrice d'adjacence
        config     : configuration globale
        n_steps    : nombre de pas de simulation
        use_damping: si True, applique l'amortissement cyclique
        node_dim   : dimension des représentations par nœud

    Returns:
        dict avec 'free_energy', 'order_param', 'epsilon_norm' par pas de temps
    """
    n_nodes = adj.shape[0]

    # Configuration avec ou sans amortissement
    config_sim = SNNConfig()
    if not use_damping:
        config_sim.cycle_damping_lambda = 0.0  # pas d'amortissement

    model = PCGNN(n_nodes, node_dim, config_sim)

    # Observations aléatoires (n_nodes premiers = observés)
    x = torch.randn(n_nodes, node_dim)
    obs_mask = torch.zeros(n_nodes, dtype=torch.bool)
    obs_mask[:n_nodes // 3] = True  # premier tiers = observé

    # Arêtes au format COO
    edge_index = adj.nonzero(as_tuple=False).t()  # (2, n_edges)

    history = {
        'free_energy': [],
        'order_param': [],
        'epsilon_norm': [],
    }

    phases = torch.rand(n_nodes) * 2 * 3.14159

    for step in range(n_steps):
        with torch.no_grad():
            out = model(x, edge_index, obs_mask)

        # Énergie libre
        F_val = out['free_energy'].item()

        # Mettre à jour les phases avec Kuramoto
        W_phase = adj * 0.01
        phases = kuramoto_update(phases, W_phase, dt=config.dt)

        # Paramètre d'ordre
        r = compute_order_parameter(phases)

        # Norme de l'erreur
        eps_norm = out['epsilon'].norm().item()

        history['free_energy'].append(F_val)
        history['order_param'].append(r)
        history['epsilon_norm'].append(eps_norm)

    return history


def run_convergence_test(
    save_dir: str = '../results',
    n_nodes: int = 10,
    n_steps: int = 80,
    node_dim: int = 16,
) -> dict:
    """
    Test de convergence complet sur les 4 topologies de graphe.

    Args:
        save_dir  : répertoire de sauvegarde des figures
        n_nodes   : nombre de nœuds par graphe
        n_steps   : nombre de pas de simulation
        node_dim  : dimension des représentations

    Returns:
        résultats : dict avec les historiques par topologie et méthode
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = SNNConfig()

    # ── Définition des topologies ────────────────────────────────────────────
    topologies = {
        'DAG (acyclique)': make_dag(n_nodes, branching=0.4),
        'Cycles simples (3-5)': make_cyclic_graph(n_nodes, n_extra_edges=3),
        'Erdős–Rényi p=0.3': make_erdos_renyi(n_nodes, p=0.3),
        'Erdős–Rényi p=0.6': make_erdos_renyi(n_nodes, p=0.6),
    }

    all_results = {}

    print("═" * 60)
    print("Test de convergence sur graphes cycliques")
    print(f"n_nodes={n_nodes}, n_steps={n_steps}, node_dim={node_dim}")
    print("═" * 60)

    for topo_name, adj in topologies.items():
        n_edges = adj.sum().item()
        n_cycles = _count_cycles_approx(adj)

        print(f"\n{topo_name}")
        print(f"  Arêtes : {n_edges:.0f} | Cycles ≈ {n_cycles}")

        # Vérification de la condition de Kuramoto
        W_kuramoto = adj * 0.01
        converge, margin = kuramoto_condition(W_kuramoto, gamma=config.gamma_stability)
        print(f"  Condition Kuramoto : {'✓ OK' if converge else '✗ FAIL'} (marge = {margin:.4f})")

        # Simulation avec amortissement
        print("  Simulation avec amortissement...")
        hist_with = simulate_convergence(adj, config, n_steps, use_damping=True, node_dim=node_dim)

        # Simulation sans amortissement
        print("  Simulation sans amortissement...")
        hist_without = simulate_convergence(adj, config, n_steps, use_damping=False, node_dim=node_dim)

        all_results[topo_name] = {
            'with_damping': hist_with,
            'without_damping': hist_without,
            'adj': adj,
            'n_edges': n_edges,
            'n_cycles': n_cycles,
            'kuramoto_ok': converge,
        }

        final_F_with = hist_with['free_energy'][-1]
        final_F_without = hist_without['free_energy'][-1]
        print(f"  F final (avec)    : {final_F_with:.4f}")
        print(f"  F final (sans)    : {final_F_without:.4f}")

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\nGénération des visualisations...")
    _plot_convergence(all_results, save_path, n_steps)

    print(f"\nFigures sauvegardées dans : {save_path}")

    return all_results


def _count_cycles_approx(adj: torch.Tensor) -> int:
    """Compte approximativement les cycles via la formule : |E| - |V| + C."""
    n = adj.shape[0]
    n_edges = adj.sum().item()
    # Pour un graphe non-orienté : cycles ≈ max(0, n_edges - n + 1)
    return max(0, int(n_edges - n + 1))


def _plot_convergence(results: dict, save_path: Path, n_steps: int) -> None:
    """Génère les figures de comparaison de convergence."""

    n_topos = len(results)
    fig, axes = plt.subplots(2, n_topos, figsize=(5 * n_topos, 8), squeeze=False)
    fig.suptitle('Convergence PC sur graphes cycliques\n(avec vs sans amortissement)',
                 fontsize=13, fontweight='bold')

    steps = list(range(1, n_steps + 1))
    colors_with = 'steelblue'
    colors_without = 'tomato'

    for col, (topo_name, data) in enumerate(results.items()):
        hist_w = data['with_damping']
        hist_wo = data['without_damping']

        # Sous-figure haut : énergie libre
        ax_F = axes[0][col]
        ax_F.plot(steps, hist_w['free_energy'], color=colors_with, linewidth=2,
                  label='Avec amortissement')
        ax_F.plot(steps, hist_wo['free_energy'], color=colors_without, linewidth=2,
                  linestyle='--', label='Sans amortissement')
        ax_F.set_title(f'{topo_name}\n(cycles ≈ {data["n_cycles"]})', fontsize=9)
        ax_F.set_xlabel('Pas de simulation')
        if col == 0:
            ax_F.set_ylabel('Énergie libre F')
        ax_F.legend(fontsize=7)

        # Icône de convergence
        ok_str = '✓ Kuramoto OK' if data['kuramoto_ok'] else '✗ Kuramoto FAIL'
        ax_F.text(0.05, 0.95, ok_str, transform=ax_F.transAxes, fontsize=7,
                  verticalalignment='top',
                  color='green' if data['kuramoto_ok'] else 'red')

        # Sous-figure bas : paramètre d'ordre r
        ax_r = axes[1][col]
        ax_r.plot(steps, hist_w['order_param'], color=colors_with, linewidth=2,
                  label='Avec amortissement')
        ax_r.plot(steps, hist_wo['order_param'], color=colors_without, linewidth=2,
                  linestyle='--', label='Sans amortissement')
        ax_r.set_xlabel('Pas de simulation')
        if col == 0:
            ax_r.set_ylabel('Paramètre d\'ordre r')
        ax_r.set_ylim(0, 1.05)
        ax_r.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax_r.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path / 'convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure supplémentaire : visualisation des graphes ────────────────────
    fig2, axes2 = plt.subplots(1, n_topos, figsize=(5 * n_topos, 4))
    if n_topos == 1:
        axes2 = [axes2]

    for col, (topo_name, data) in enumerate(results.items()):
        adj_np = data['adj'].numpy()
        G = nx.from_numpy_array(adj_np, create_using=nx.DiGraph)
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx(G, pos=pos, ax=axes2[col],
                        node_color='lightblue', node_size=300,
                        font_size=7, arrows=True,
                        edge_color='gray', alpha=0.8)
        axes2[col].set_title(topo_name, fontsize=9)
        axes2[col].axis('off')

    plt.suptitle('Topologies de graphes testées', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path / 'graph_topologies.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ convergence_comparison.png")
    print(f"  ✓ graph_topologies.png")


if __name__ == '__main__':
    results = run_convergence_test(
        save_dir='../results',
        n_nodes=10,
        n_steps=80,
        node_dim=16,
    )
