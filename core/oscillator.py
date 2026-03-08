"""
core/oscillator.py — Horloge oscillatoire thêta/gamma pour la coordination temporelle.

Le couplage thêta-gamma est un mécanisme neurophysiologique bien documenté
dans l'hippocampe et le cortex préfrontal. Il organise l'information en séquences :
- Chaque cycle thêta (~167ms) contient plusieurs sous-cycles gamma (~25ms)
- L'amplitude gamma est maximale à la phase thêta optimale (≈ pic thêta)
- Ce couplage permet un multiplexage temporel : différents "items" sémantiques
  sont représentés dans différents sous-cycles gamma d'un même cycle thêta

Mathématiquement :
    φ_θ(t) = 2π · f_θ · t
    φ_γ(t) = 2π · f_γ · t
    A_γ(t) = 0.5 · (1 + cos(φ_θ(t)))   ← modulation en amplitude

Référence : Lisman & Jensen (2013) "The Theta-Gamma Neural Code"
"""

import math
import torch
from config import SNNConfig


class OscillatoryClock:
    """
    Horloge oscillatoire maintenant les phases thêta et gamma couplées.

    Le couplage est unidirectionnel (thêta → gamma) : la phase thêta
    module l'amplitude des oscillations gamma sans affecter sa fréquence.
    Cela crée des fenêtres temporelles d'excitabilité accrue.

    États internes :
    - _time    : temps courant en ms
    - _phi_theta : phase thêta courante ∈ [0, 2π)
    - _phi_gamma : phase gamma courante ∈ [0, 2π)
    """

    def __init__(self, config: SNNConfig):
        self.config = config
        self._time: float = 0.0
        self._phi_theta: float = 0.0
        self._phi_gamma: float = 0.0

        # Incrément de phase par pas dt (converti Hz → radians/ms)
        # Δφ = 2π · f[Hz] · dt[ms] · 1e-3[s/ms]
        self._dphi_theta = 2.0 * math.pi * config.theta_freq * config.dt * 1e-3
        self._dphi_gamma = 2.0 * math.pi * config.gamma_freq * config.dt * 1e-3

    def step(self) -> dict[str, float]:
        """
        Avance l'horloge d'un pas dt et retourne les phases/amplitudes courantes.

        Returns:
            dict avec les clés :
            - 'time'            : temps courant en ms
            - 'phi_theta'       : phase thêta ∈ [0, 2π)
            - 'phi_gamma'       : phase gamma ∈ [0, 2π)
            - 'amplitude_gamma' : amplitude gamma modulée ∈ [0, 1]
            - 'theta_peak'      : True si on est au pic thêta (φ_θ ≈ 0 mod 2π)
        """
        self._time += self.config.dt

        # Mise à jour des phases (modulo 2π pour rester dans [0, 2π))
        self._phi_theta = (self._phi_theta + self._dphi_theta) % (2.0 * math.pi)
        self._phi_gamma = (self._phi_gamma + self._dphi_gamma) % (2.0 * math.pi)

        # Amplitude gamma : modulation cosinus par la phase thêta
        # A_γ = 0.5 · (1 + cos(φ_θ)) ∈ [0, 1]
        # Maximum (1.0) quand φ_θ = 0, minimum (0.0) quand φ_θ = π
        amplitude_gamma = 0.5 * (1.0 + math.cos(self._phi_theta))

        # Détection du pic thêta (transition 2π → 0)
        theta_peak = self._phi_theta < self._dphi_theta

        return {
            'time': self._time,
            'phi_theta': self._phi_theta,
            'phi_gamma': self._phi_gamma,
            'amplitude_gamma': amplitude_gamma,
            'theta_peak': theta_peak,
        }

    def get_theta_phase(self) -> float:
        """Retourne la phase thêta courante ∈ [0, 2π)."""
        return self._phi_theta

    def get_gamma_phase(self) -> float:
        """Retourne la phase gamma courante ∈ [0, 2π)."""
        return self._phi_gamma

    def get_gamma_amplitude(self) -> float:
        """
        Amplitude courante des oscillations gamma, modulée par thêta.
        ∈ [0, 1] — 1 au pic thêta, 0 au creux thêta.
        """
        return 0.5 * (1.0 + math.cos(self._phi_theta))

    def reset(self, initial_phase_theta: float = 0.0, initial_phase_gamma: float = 0.0) -> None:
        """
        Réinitialise l'horloge avec des phases initiales optionnelles.

        Args:
            initial_phase_theta : phase thêta initiale ∈ [0, 2π)
            initial_phase_gamma : phase gamma initiale ∈ [0, 2π)
        """
        self._time = 0.0
        self._phi_theta = initial_phase_theta % (2.0 * math.pi)
        self._phi_gamma = initial_phase_gamma % (2.0 * math.pi)

    @property
    def time_ms(self) -> float:
        """Temps courant en millisecondes."""
        return self._time

    def gamma_cycle_index(self) -> int:
        """
        Index du sous-cycle gamma courant dans le cycle thêta.

        Utilisé pour le multiplexage temporel : différents items sémantiques
        occupent différents indices gamma (0, 1, ..., n_gamma_per_theta - 1).
        """
        ratio = self.config.gamma_freq / self.config.theta_freq
        # Position dans le cycle thêta courant en unités de cycles gamma
        theta_progress = self._phi_theta / (2.0 * math.pi)
        return int(theta_progress * ratio) % int(ratio)
