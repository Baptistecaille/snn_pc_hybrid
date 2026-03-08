"""
training/viz.py — Visualisations pour le suivi de l'entraînement SNN-PC.

Toutes les fonctions :
- Retournent un objet plt.Figure (pour les notebooks)
- Sauvegardent automatiquement si save_path est fourni
- Sont autonomes (pas de dépendance aux modules du projet)

Fonctions disponibles :
    plot_raster(spikes_W, spikes_B, theta_phases, ...)
        → Raster plot double panneau avec phase thêta

    plot_training_dashboard(log_csv_path, ...)
        → Dashboard 2×2 chargé depuis le CSV de logging
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def plot_raster(
    spikes_W: torch.Tensor,
    spikes_B: torch.Tensor,
    theta_phases: list,
    title: str = 'Raster plot — Wernicke & Broca',
    save_path: str = None,
) -> plt.Figure:
    """
    Raster plot double panneau (Wernicke haut, Broca bas).

    Panneau du bas : phase thêta au cours du temps, comme référence oscillatoire.
    Les spikes sont visualisés avec des traits verticaux (scatter marker='|').

    Args:
        spikes_W      : (T, n_W) — spikes binaires du module Wernicke
        spikes_B      : (T, n_B) — spikes binaires du module Broca
        theta_phases  : liste de T valeurs de phase thêta ∈ [0, 2π)
        title         : titre de la figure
        save_path     : chemin de sauvegarde (PNG), ou None

    Returns:
        fig : objet matplotlib.Figure
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 3, 1]},
        sharex=True,
    )

    T = spikes_W.shape[0]
    t = np.arange(T)

    # ── Panneau Wernicke ──────────────────────────────────────────────────────
    ax_W = axes[0]
    s_W = spikes_W.cpu() if hasattr(spikes_W, 'cpu') else torch.tensor(spikes_W)
    spike_times_W, neuron_ids_W = torch.where(s_W > 0.5)
    if len(spike_times_W) > 0:
        ax_W.scatter(
            spike_times_W.numpy(), neuron_ids_W.numpy(),
            s=2, c='steelblue', alpha=0.7, marker='|', linewidths=1.2,
        )
    ax_W.set_ylabel('Neurone (Wernicke)', fontsize=9)
    ax_W.set_xlim(0, T)
    ax_W.set_ylim(-0.5, spikes_W.shape[1] - 0.5)
    ax_W.set_title(title, fontsize=11, fontweight='bold')
    ax_W.set_facecolor('#f0f4ff')
    # Taux moyen
    rate_W = float(s_W.float().mean().item())
    ax_W.text(0.02, 0.92, f'Taux moy. : {rate_W:.4f} sp/pas',
              transform=ax_W.transAxes, fontsize=7, color='steelblue')

    # ── Panneau Broca ─────────────────────────────────────────────────────────
    ax_B = axes[1]
    s_B = spikes_B.cpu() if hasattr(spikes_B, 'cpu') else torch.tensor(spikes_B)
    spike_times_B, neuron_ids_B = torch.where(s_B > 0.5)
    if len(spike_times_B) > 0:
        ax_B.scatter(
            spike_times_B.numpy(), neuron_ids_B.numpy(),
            s=2, c='firebrick', alpha=0.7, marker='|', linewidths=1.2,
        )
    ax_B.set_ylabel('Neurone (Broca)', fontsize=9)
    ax_B.set_xlim(0, T)
    ax_B.set_ylim(-0.5, spikes_B.shape[1] - 0.5)
    ax_B.set_facecolor('#fff0f0')
    rate_B = float(s_B.float().mean().item())
    ax_B.text(0.02, 0.92, f'Taux moy. : {rate_B:.4f} sp/pas',
              transform=ax_B.transAxes, fontsize=7, color='firebrick')

    # ── Phase thêta ───────────────────────────────────────────────────────────
    ax_theta = axes[2]
    theta_arr = np.array(theta_phases[:T])
    ax_theta.plot(t[:len(theta_arr)], theta_arr, color='darkorange', lw=1.2)
    ax_theta.axhline(y=np.pi, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_theta.set_ylabel('φ_θ (rad)', fontsize=9)
    ax_theta.set_xlabel('Pas de simulation (dt)', fontsize=9)
    ax_theta.set_xlim(0, T)
    ax_theta.set_ylim(0, 2 * np.pi)
    ax_theta.set_yticks([0, np.pi, 2 * np.pi])
    ax_theta.set_yticklabels(['0', 'π', '2π'], fontsize=7)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Raster plot sauvegardé : {save_path}")

    return fig


def plot_training_dashboard(
    log_csv_path: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Dashboard 2×2 chargé depuis le fichier CSV de training_log.

    Layout :
        [0,0] Énergie libre F_W (bleu) et F_B (rouge) au fil des steps
        [0,1] Ordre de Kuramoto r_W et r_B + ligne de seuil r=0.7
        [1,0] Loss totale (F_W + F_B + sync) par phase
        [1,1] Learning rate schedule (échelle log)

    Les transitions de phase sont marquées par des lignes verticales colorées.
    Chaque courbe est lissée par moyenne mobile (window=100 steps).

    Args:
        log_csv_path : chemin vers le fichier CSV produit par CurriculumTrainer
        save_path    : chemin de sauvegarde (PNG), ou None

    Returns:
        fig : objet matplotlib.Figure

    Raises:
        ImportError  : si pandas n'est pas installé
        FileNotFoundError : si log_csv_path n'existe pas
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas est requis pour le dashboard : pip install pandas>=2.1")

    if not Path(log_csv_path).exists():
        raise FileNotFoundError(f"Fichier de log introuvable : {log_csv_path}")

    df = pd.read_csv(log_csv_path)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

    # Couleurs par phase
    phases     = df['phase'].unique().tolist()
    n_phases   = len(phases)
    colors_ph  = plt.cm.Set2(np.linspace(0, 1, max(n_phases, 1)))

    # Indices de début de chaque phase (pour les lignes verticales)
    boundaries = [df[df['phase'] == p].index[0] for p in phases]

    def moving_avg(series, w: int = 100) -> "pd.Series":
        return series.rolling(window=w, min_periods=1).mean()

    def add_phase_lines(ax) -> None:
        for i, b in enumerate(boundaries):
            ax.axvline(b, color=colors_ph[i], lw=1.0, ls='--', alpha=0.6)

    # ── [0,0] Énergie libre ────────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    if 'F_W' in df.columns:
        ax00.plot(moving_avg(df['F_W']), color='steelblue', lw=1.5, label='F_W (Wernicke)')
    if 'F_B' in df.columns:
        ax00.plot(moving_avg(df['F_B']), color='firebrick', lw=1.5, label='F_B (Broca)')
    add_phase_lines(ax00)
    ax00.set_title('Énergie libre F(t)', fontweight='bold')
    ax00.set_ylabel('F'); ax00.legend(fontsize=8)
    ax00.set_xlabel('Step')

    # ── [0,1] Ordre de Kuramoto ────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    if 'r_W' in df.columns:
        ax01.plot(moving_avg(df['r_W']), color='steelblue', lw=1.5, label='r_W')
    if 'r_B' in df.columns:
        ax01.plot(moving_avg(df['r_B']), color='firebrick', lw=1.5, label='r_B')
    ax01.axhline(0.7, color='green', lw=1.0, ls=':', alpha=0.8, label='seuil r=0.70')
    ax01.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.4)
    add_phase_lines(ax01)
    ax01.set_title('Ordre de Kuramoto r(t)', fontweight='bold')
    ax01.set_ylim(0, 1.05)
    ax01.set_ylabel('r'); ax01.legend(fontsize=8)
    ax01.set_xlabel('Step')

    # ── [1,0] Loss totale ──────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    if 'F_total' in df.columns:
        ax10.plot(moving_avg(df['F_total']), color='purple', lw=1.5, label='F total')
    if 'sync_loss' in df.columns:
        ax10.plot(moving_avg(df['sync_loss']), color='darkorange', lw=1.0,
                  ls='--', alpha=0.7, label='L_sync')
    add_phase_lines(ax10)
    ax10.set_title('Loss totale et terme de synchronisation', fontweight='bold')
    ax10.set_ylabel('Loss'); ax10.set_xlabel('Step')
    ax10.legend(fontsize=8)

    # ── [1,1] Learning rate ────────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    if 'lr' in df.columns:
        ax11.semilogy(df['lr'], color='darkorange', lw=1.2, alpha=0.8)
    add_phase_lines(ax11)
    ax11.set_title('Learning rate schedule', fontweight='bold')
    ax11.set_ylabel('LR (échelle log)'); ax11.set_xlabel('Step')

    # ── Légende des phases ─────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=colors_ph[i], label=p, alpha=0.8)
        for i, p in enumerate(phases)
    ]
    fig.legend(
        handles=legend_patches, loc='lower center',
        ncol=n_phases, fontsize=9, title='Phases du curriculum',
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle(
        'Tableau de bord — Entraînement SNN-PC (Wikipedia FR + OSCAR)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Dashboard sauvegardé : {save_path}")

    return fig


def plot_phase_curriculum_summary(
    results: dict,
    save_path: str = None,
) -> plt.Figure:
    """
    Graphique récapitulatif des métriques finales par phase.

    Affiche F_final et r_final pour chaque phase sous forme de barres,
    avec les lignes de seuil correspondantes (config.phase_thresholds).

    Args:
        results   : dict phase → {'F_final', 'r_final', 'steps'} (retour de trainer.run())
        save_path : chemin de sauvegarde (PNG), ou None

    Returns:
        fig : objet matplotlib.Figure
    """
    phases  = list(results.keys())
    F_vals  = [results[p]['F_final'] for p in phases]
    r_vals  = [results[p]['r_final'] for p in phases]
    steps   = [results[p]['steps']   for p in phases]
    x       = np.arange(len(phases))
    width   = 0.35

    fig, (ax_F, ax_r) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Énergie libre ─────────────────────────────────────────────────────────
    bars_F = ax_F.bar(x, F_vals, width, color='steelblue', alpha=0.7, label='F final')
    ax_F.set_xticks(x)
    ax_F.set_xticklabels(phases, rotation=20, ha='right', fontsize=8)
    ax_F.set_title('Énergie libre finale par phase', fontweight='bold')
    ax_F.set_ylabel('F')
    for i, (bar, v) in enumerate(zip(bars_F, F_vals)):
        ax_F.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    # ── Ordre de Kuramoto ─────────────────────────────────────────────────────
    bars_r = ax_r.bar(x, r_vals, width, color='darkorange', alpha=0.7, label='r final')
    ax_r.axhline(y=0.7, color='green', lw=1.0, ls='--', alpha=0.7, label='seuil r=0.7')
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(phases, rotation=20, ha='right', fontsize=8)
    ax_r.set_title('Ordre de Kuramoto final par phase', fontweight='bold')
    ax_r.set_ylabel('r')
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(fontsize=8)
    for i, (bar, v) in enumerate(zip(bars_r, r_vals)):
        ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Récapitulatif du curriculum', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Récapitulatif sauvegardé : {save_path}")

    return fig
