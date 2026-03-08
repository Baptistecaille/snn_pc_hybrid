"""
graph/phase_sync.py — Convergence de type Kuramoto pour le Predictive Coding sur graphes.

Le modèle de Kuramoto décrit la synchronisation d'oscillateurs couplés :
    dφ_i/dt = ω_i + K/N · Σ_j sin(φ_j - φ_i)

Pour notre framework SNN-PC, nous utilisons une généralisation sur graphes
arbitraires avec des poids de couplage asymétriques.

Condition de convergence (théorème de Kuramoto étendu) :
    γ > σ_max(W) · L

où :
- γ    : paramètre de stabilité (doit dominer le couplage)
- σ_max(W) : plus grande valeur singulière de la matrice de couplage W
- L    : constante de Lipschitz du couplage de phase (≤ 1 pour sin)

Sur un graphe cyclique, la cohérence de phase doit satisfaire :
    Σ_{(i,j) ∈ C} Δφ_ij ≡ 0 (mod 2π)
pour tout cycle C. Toute violation indique une incohérence structurelle.

Références :
- Strogatz (2000) "From Kuramoto to Crawford: exploring the onset of synchronization"
- Lohmiller & Slotine (1998) "On contraction analysis for non-linear systems"
"""

import math
import torch
import numpy as np
from typing import Optional


def kuramoto_condition(
    W: torch.Tensor,
    gamma: float,
    lipschitz_constant: float = 1.0,
) -> tuple[bool, float]:
    """
    Vérifie la condition de convergence de Kuramoto pour un graphe de couplage.

    Condition : γ > σ_max(W) · L

    La condition garantit que le système d'oscillateurs couplés converge
    vers un état synchronisé (ou au moins borné) malgré les cycles.

    La justification vient de la théorie de contraction (Lohmiller & Slotine) :
    si le Jacobien du système est uniformément négatif défini, le système converge.
    Pour le couplage sinusoïdal, L = max |d/dφ sin(φ)| = 1.

    Args:
        W                 : (n, n) — matrice de couplage de phase
        gamma             : paramètre de stabilité γ > 0
        lipschitz_constant: constante de Lipschitz L du couplage (≤ 1 pour sin)

    Returns:
        converge : True si la condition est satisfaite
        margin   : γ - σ_max(W)·L (positif → convergence garantie)

    Exemple:
        >>> W = torch.randn(5, 5) * 0.1
        >>> converge, margin = kuramoto_condition(W, gamma=0.1)
        >>> print(f"Convergence: {converge}, marge: {margin:.4f}")
    """
    # Calcul des valeurs singulières via SVD
    # σ_max(W) = plus grande valeur singulière
    singular_values = torch.linalg.svdvals(W)
    sigma_max = singular_values.max().item()

    # Vérification de la condition
    threshold = sigma_max * lipschitz_constant
    margin = gamma - threshold
    converge = margin > 0

    return converge, margin


def compute_order_parameter(phases: torch.Tensor) -> float:
    """
    Calcule le paramètre d'ordre de Kuramoto.

    r = |1/N · Σ_k e^{i·φ_k}|

    r ≈ 1 → synchronisation totale (tous les oscillateurs ont la même phase)
    r ≈ 0 → désynchronisation complète (phases uniformément distribuées)

    Ce paramètre est la mesure standard de cohérence de phase dans les
    systèmes d'oscillateurs couplés.

    Args:
        phases : (n,) ou (batch, n) — phases des oscillateurs ∈ [0, 2π)

    Returns:
        r : paramètre d'ordre ∈ [0, 1]

    Exemple:
        >>> phases = torch.zeros(10)  # tous synchronisés → r = 1
        >>> r = compute_order_parameter(phases)
        >>> assert abs(r - 1.0) < 1e-6
    """
    if phases.dim() == 1:
        phases = phases.unsqueeze(0)  # (1, n)

    # Représentation complexe des phases sur le cercle unitaire
    # e^{iφ} = cos(φ) + i·sin(φ)
    complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))  # (batch, n)

    # Moyenne des vecteurs complexes
    mean_complex = complex_phases.mean(dim=-1)  # (batch,)

    # Module du vecteur moyen = paramètre d'ordre
    r = mean_complex.abs().mean().item()
    return float(r)


def phase_consistency_check(
    graph: torch.Tensor,
    phases: torch.Tensor,
) -> dict[str, torch.Tensor | list | float]:
    """
    Vérifie la cohérence de phase sur tous les cycles détectés dans le graphe.

    Pour chaque cycle C = (v_0, v_1, ..., v_k, v_0) dans le graphe :
        Σ_{j=0}^{k} Δφ_{v_j, v_{j+1}} ≡ 0 (mod 2π)

    Cette condition est analogue à la loi des mailles de Kirchhoff.
    Une violation indique une frustration de phase (contradiction structurelle).

    Algorithme : DFS pour trouver tous les cycles fondamentaux (base de cycles).
    Pour un graphe non-orienté, la base de cycles a |E| - |V| + 1 éléments.

    Args:
        graph  : (n, n) — matrice d'adjacence (1 si arête, 0 sinon)
        phases : (n,)   — phases courantes des oscillateurs ∈ [0, 2π)

    Returns:
        dict avec :
        - 'cycles'      : liste des cycles détectés (chacun = liste d'indices)
        - 'consistency' : (n_cycles,) — score de cohérence par cycle ∈ [0, 1]
        - 'global_sync' : float — paramètre d'ordre global r ∈ [0, 1]
    """
    n = graph.shape[0]
    adj = graph.cpu().numpy()
    phases_np = phases.cpu().numpy()

    # ── Détection des cycles via DFS (algorithme de Johnson) ─────────────────
    cycles = _find_cycles_dfs(adj, n)

    # ── Calcul de la cohérence par cycle ──────────────────────────────────────
    consistency_scores = []
    for cycle in cycles:
        # Somme des différences de phase le long du cycle
        phase_sum = 0.0
        for i in range(len(cycle)):
            j = (i + 1) % len(cycle)
            delta_phi = phases_np[cycle[j]] - phases_np[cycle[i]]
            phase_sum += delta_phi

        # La somme devrait être ≡ 0 (mod 2π) pour un système cohérent
        # Score : 1 si |phase_sum mod 2π| < ε, décroît vers 0 sinon
        residual = abs(phase_sum % (2 * math.pi))
        if residual > math.pi:
            residual = 2 * math.pi - residual  # distance angulaire minimale
        consistency = math.exp(-residual)  # ∈ (0, 1]

        consistency_scores.append(consistency)

    # ── Paramètre d'ordre global ───────────────────────────────────────────────
    global_sync = compute_order_parameter(phases)

    return {
        'cycles': cycles,
        'consistency': torch.tensor(consistency_scores) if consistency_scores else torch.zeros(0),
        'global_sync': global_sync,
    }


def _find_cycles_dfs(adj: np.ndarray, n: int, max_cycles: int = 50) -> list[list[int]]:
    """
    Trouve les cycles fondamentaux d'un graphe par DFS.

    Pour éviter une explosion combinatoire sur les graphes denses,
    on se limite aux max_cycles premiers cycles trouvés.

    Args:
        adj       : (n, n) matrice d'adjacence numpy
        n         : nombre de nœuds
        max_cycles: nombre maximal de cycles à retourner

    Returns:
        cycles : liste de cycles, chacun = liste d'indices de nœuds
    """
    cycles = []
    visited = [False] * n
    parent = [-1] * n

    def dfs(v: int, path: list[int]) -> None:
        if len(cycles) >= max_cycles:
            return
        visited[v] = True
        path.append(v)

        for u in range(n):
            if adj[v][u] == 0:
                continue
            if len(cycles) >= max_cycles:
                break
            if not visited[u]:
                parent[u] = v
                dfs(u, path)
            elif u != parent[v] and u in path:
                # Cycle détecté : extraire le cycle depuis path
                cycle_start = path.index(u)
                cycle = path[cycle_start:]
                if len(cycle) >= 3:
                    cycles.append(cycle[:])

        path.pop()

    for start in range(n):
        if not visited[start] and len(cycles) < max_cycles:
            dfs(start, [])

    return cycles


def kuramoto_update(
    phases: torch.Tensor,
    W: torch.Tensor,
    dt: float,
    natural_freq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Met à jour les phases d'oscillateurs selon la dynamique de Kuramoto.

    dφ_i/dt = ω_i + Σ_j W_ij · sin(φ_j - φ_i)

    Discrétisation Euler explicite :
    φ_i(t+dt) = φ_i(t) + dt · [ω_i + Σ_j W_ij · sin(φ_j - φ_i)]

    Args:
        phases      : (n,) — phases courantes ∈ [0, 2π)
        W           : (n, n) — matrice de couplage (W_ij = force i→j)
        dt          : pas temporel (en unités consistantes avec ω)
        natural_freq: (n,) — fréquences naturelles ω_i (défaut : toutes égales à 0)

    Returns:
        phases_new : (n,) — phases mises à jour, remises dans [0, 2π)
    """
    n = phases.shape[0]

    if natural_freq is None:
        natural_freq = torch.zeros(n, device=phases.device)

    # Différences de phase entre toutes les paires : δφ_ji = φ_j - φ_i
    delta_phi = phases.unsqueeze(0) - phases.unsqueeze(1)  # (n, n)

    # Terme de couplage : Σ_j W_ij · sin(φ_j - φ_i)
    coupling = (W * torch.sin(delta_phi)).sum(dim=1)  # (n,)

    # Mise à jour Euler
    dphi_dt = natural_freq + coupling
    phases_new = phases + dt * dphi_dt

    # Ramener dans [0, 2π)
    phases_new = phases_new % (2.0 * math.pi)

    return phases_new
