"""
config.py — Hyperparamètres globaux du framework SNN-PC hybride.

Tous les hyperparamètres sont regroupés dans une dataclass SNNConfig
pour faciliter la reproductibilité et le logging des expériences.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SNNConfig:
    # ── Dynamique membranaire ────────────────────────────────────────────────
    tau_m: float = 20.0         # ms — constante de temps membranaire (intégration des courants)
    tau_syn: float = 5.0        # ms — constante de temps synaptique (décroissance du courant)
    v_rest: float = -70.0       # mV — potentiel de repos
    v_threshold: float = -55.0  # mV — seuil de décharge (dépolarisation)
    v_reset: float = -75.0      # mV — potentiel post-spike (hyperpolarisation)
    dt: float = 0.1             # ms — pas de discrétisation temporelle

    # ── Oscillations thêta/gamma ─────────────────────────────────────────────
    # Couplage thêta-gamma : la phase thêta module l'amplitude gamma,
    # ce qui crée des fenêtres temporelles pour le binding sémantique.
    theta_freq: float = 6.0     # Hz — fréquence thêta (horloge globale, ~166ms/cycle)
    gamma_freq: float = 40.0    # Hz — fréquence gamma (binding local, ~25ms/cycle)

    # ── Predictive Coding ─────────────────────────────────────────────────────
    # η contrôle l'influence des erreurs de prédiction sur le potentiel membranaire.
    # sigma_prior = écart-type du prior gaussien ; précision = 1/sigma².
    eta_pc: float = 0.1             # poids du terme d'erreur PC dans la dynamique neuronale
    sigma_prior: float = 1.0        # variance du prior (précision = 1/sigma²)
    n_inference_steps: int = 20     # itérations d'inférence PC par pas temporel

    # ── STDP (Spike-Timing-Dependent Plasticity) ──────────────────────────────
    # La règle STDP est compatible avec PC : A_plus potentialise si pré→post,
    # A_minus déprime si post→pré. Les fenêtres tau_plus/minus définissent
    # la sensibilité temporelle de la plasticité.
    A_plus: float = 0.01            # amplitude de potentialisation (pre avant post)
    A_minus: float = 0.012          # amplitude de dépression (post avant pre)
    tau_plus: float = 20.0          # ms — fenêtre temporelle pour A_plus
    tau_minus: float = 20.0         # ms — fenêtre temporelle pour A_minus
    lr_weights: float = 1e-3        # taux d'apprentissage pour les poids synaptiques

    # ── Architecture des modules ──────────────────────────────────────────────
    dim_wernicke: int = 128         # dimension de la représentation sémantique
    dim_broca: int = 128            # dimension de la représentation syntaxique
    dim_arcuate: int = 64           # dimension du canal fascicule arqué

    # ── Convergence sur graphes cycliques ─────────────────────────────────────
    # Condition de Kuramoto étendue : γ > σ_max(W) · L
    # γ doit être strictement supérieur au produit de la valeur singulière
    # maximale de W et de la constante de Lipschitz L du couplage de phase.
    gamma_stability: float = 0.1    # paramètre de stabilité γ
    cycle_damping_lambda: float = 0.5  # λ pour l'amortissement cyclique : f(n) = exp(-λ·n)

    # ── Device ────────────────────────────────────────────────────────────────
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── Curriculum d'entraînement ─────────────────────────────────────────────
    # 5 phases progressives : bootstrap → wikipedia → OSCAR
    # Chaque phase a ses propres critères de passage (F_max, r_min, steps).
    curriculum_phases: list = field(default_factory=lambda: [
        'bootstrap', 'wikipedia_short', 'wikipedia_long',
        'oscar_filtered', 'oscar_full',
    ])

    # Critères de passage par phase (None = pas de limite)
    # F_max : énergie libre maximale tolérée (moyenne sur 100 derniers steps)
    # r_min : ordre de Kuramoto minimal requis
    # steps : nombre de steps minimum avant de pouvoir passer
    phase_thresholds: dict = field(default_factory=lambda: {
        'bootstrap':       {'F_max': 2.0,  'r_min': 0.50, 'steps': 1_000},
        'wikipedia_short': {'F_max': 1.5,  'r_min': 0.60, 'steps': 10_000},
        'wikipedia_long':  {'F_max': 1.2,  'r_min': 0.65, 'steps': 50_000},
        'oscar_filtered':  {'F_max': 1.0,  'r_min': 0.70, 'steps': 100_000},
        'oscar_full':      {'F_max': None, 'r_min': 0.70, 'steps': None},
    })

    # Longueur max de séquence par phase (tokens).
    # Augmentation progressive : crucial pour les SNNs dont la mémoire
    # de travail est bornée par le nombre de cycles gamma.
    phase_max_tokens: dict = field(default_factory=lambda: {
        'bootstrap':       16,
        'wikipedia_short': 64,
        'wikipedia_long':  256,
        'oscar_filtered':  256,
        'oscar_full':      512,
    })

    # ── Tokenizer et chemins ──────────────────────────────────────────────────
    tokenizer_name: str  = 'camembert-base'   # BPE français (32k vocab)
    data_cache_dir: str  = './cache'
    checkpoint_dir: str  = './checkpoints'
    log_csv_path: str    = './logs/training_log.csv'
    wikipedia_config_name: str = '20231101.fr'
    wiki_max_articles: Optional[int] = None
    oscar_buffer_size: int = 10_000
    max_steps_override: Optional[int] = None

    def __post_init__(self):
        # Vérifications de cohérence physique
        assert self.v_reset < self.v_rest < self.v_threshold, (
            "Hiérarchie membranaire violée : v_reset < v_rest < v_threshold requise"
        )
        assert self.dt > 0, "Le pas temporel dt doit être strictement positif"
        assert self.tau_m > self.tau_syn, (
            "tau_m > tau_syn requis pour la stabilité de l'intégration LIF"
        )
        assert self.gamma_freq > self.theta_freq, (
            "gamma_freq > theta_freq requis pour le couplage thêta-gamma"
        )

    @property
    def R_membrane(self) -> float:
        """Résistance membranaire déduite (R = tau_m / C_m, avec C_m = 1 par convention)."""
        return self.tau_m  # en MΩ, avec C_m = 1 nF (convention)

    @property
    def precision_prior(self) -> float:
        """Précision du prior gaussien (inverse de la variance)."""
        return 1.0 / (self.sigma_prior ** 2)

    @property
    def steps_per_theta_cycle(self) -> int:
        """Nombre de pas de simulation par cycle thêta."""
        period_ms = 1000.0 / self.theta_freq  # ms
        return int(period_ms / self.dt)

    @property
    def steps_per_gamma_cycle(self) -> int:
        """Nombre de pas de simulation par cycle gamma."""
        period_ms = 1000.0 / self.gamma_freq  # ms
        return int(period_ms / self.dt)
