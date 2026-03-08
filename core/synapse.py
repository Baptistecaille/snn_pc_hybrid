"""
core/synapse.py — Synapse avec plasticité STDP compatible Predictive Coding.

La règle STDP (Spike-Timing-Dependent Plasticity) modifie les poids synaptiques
en fonction du délai temporel entre les spikes pré- et post-synaptiques :

    Δw = A_+ · exp(-|Δt| / τ_+)   si Δt = t_post - t_pre > 0  (potentialisation)
    Δw = -A_- · exp(-|Δt| / τ_-)  si Δt = t_post - t_pre < 0  (dépression)

Compatibilité PC : La règle STDP est compatible avec la minimisation de l'énergie
libre F si on interprète la potentialisation comme l'apprentissage à réduire
les erreurs de prédiction (Rao & Ballard, 1999).

La synapse maintient des traces éligibles (eligibility traces) :
    x_pre(t)  : trace de spike pré-synaptique, decay exp. avec τ_+
    x_post(t) : trace de spike post-synaptique, decay exp. avec τ_-

Ces traces servent de mémoire à court terme pour calculer la corrélation
temporelle entre spikes.
"""

import torch
import torch.nn as nn
from config import SNNConfig


class STDPSynapse(nn.Module):
    """
    Synapse linéaire avec plasticité STDP en ligne.

    Implémente une couche linéaire dont les poids sont mis à jour à la fois
    par backpropagation (gradient de F) et par la règle STDP locale.

    En mode SNN pur, seule la règle STDP est active.
    En mode hybride PC, les deux règles coexistent avec des taux d'apprentissage
    différents : lr_weights pour le gradient, A_+/A_- pour STDP.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: SNNConfig,
        plastic: bool = True,
        delay_ms: float = 0.0,
    ):
        """
        Args:
            n_pre    : nombre de neurones pré-synaptiques
            n_post   : nombre de neurones post-synaptiques
            config   : configuration globale
            plastic  : si True, applique la règle STDP
            delay_ms : délai axonal en ms (0 = instantané)
        """
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config
        self.plastic = plastic
        self.delay_steps = max(0, int(delay_ms / config.dt))

        # Poids synaptiques — initialisés avec une distribution de Kaiming
        self.weight = nn.Parameter(
            torch.empty(n_post, n_pre).normal_(0, 1.0 / (n_pre ** 0.5))
        )

        # Facteurs de décroissance des traces (calculés une fois)
        self.alpha_pre = 1.0 - config.dt / config.tau_plus
        self.alpha_post = 1.0 - config.dt / config.tau_minus

        # Traces éligibles — buffers non-entraînables
        self.register_buffer('trace_pre', torch.zeros(1, n_pre))
        self.register_buffer('trace_post', torch.zeros(1, n_post))

        # Buffer de délai : file FIFO pour les spikes pré-synaptiques
        if self.delay_steps > 0:
            self.register_buffer(
                'delay_buffer',
                torch.zeros(self.delay_steps, 1, n_pre)
            )
            self._delay_idx = 0
        else:
            self.delay_buffer = None

    def forward(self, spikes_pre: torch.Tensor) -> torch.Tensor:
        """
        Calcule le courant post-synaptique et met à jour les traces STDP.

        Args:
            spikes_pre : (batch, n_pre) — spikes pré-synaptiques au pas courant

        Returns:
            I_post : (batch, n_post) — courant synaptique post-synaptique
        """
        batch = spikes_pre.shape[0]

        # Gestion du délai axonal (file FIFO circulaire)
        if self.delay_buffer is not None and self.delay_steps > 0:
            # Lire le spike le plus ancien (= spike retardé)
            delayed_spikes = self.delay_buffer[self._delay_idx, :, :]  # (1, n_pre)
            # Écrire les spikes courants dans le buffer
            self.delay_buffer[self._delay_idx, :, :] = spikes_pre[0:1].detach()
            self._delay_idx = (self._delay_idx + 1) % self.delay_steps
            spikes_effective = delayed_spikes.expand(batch, -1)
        else:
            spikes_effective = spikes_pre

        # Courant synaptique : I = W · s_pre
        I_post = spikes_effective @ self.weight.t()  # (batch, n_post)

        # Mise à jour des traces éligibles (pré uniquement ici, post dans update_stdp)
        self.trace_pre = self.alpha_pre * self.trace_pre + spikes_effective[0:1].detach()

        return I_post

    def update_stdp(
        self,
        spikes_pre: torch.Tensor,   # (batch, n_pre)
        spikes_post: torch.Tensor,  # (batch, n_post)
    ) -> dict[str, torch.Tensor]:
        """
        Applique la règle STDP pour mettre à jour les poids.

        Δw_ij = A_+ · x_pre_j · s_post_i  (potentialisation : pre avant post)
               - A_- · s_pre_j · x_post_i  (dépression : post avant pre)

        Args:
            spikes_pre  : spikes pré-synaptiques au pas courant
            spikes_post : spikes post-synaptiques au pas courant

        Returns:
            dict avec 'dw' (variation de poids) et 'ltp'/'ltd' (amplitudes)
        """
        if not self.plastic:
            return {'dw': torch.zeros_like(self.weight)}

        batch = spikes_pre.shape[0]
        s_pre = spikes_pre.float().mean(dim=0, keepdim=True)   # (1, n_pre)
        s_post = spikes_post.float().mean(dim=0, keepdim=True)  # (1, n_post)

        # Mise à jour de la trace post-synaptique
        self.trace_post = self.alpha_post * self.trace_post + s_post.detach()

        # Potentialisation (LTP) : corrélation trace_pre × s_post
        # Δw+ = A+ · x_pre^T ⊗ s_post  → (n_post, n_pre)
        ltp = self.config.A_plus * s_post.t() @ self.trace_pre  # (n_post, n_pre)

        # Dépression (LTD) : corrélation s_pre × trace_post
        # Δw- = -A- · trace_post^T ⊗ s_pre  → (n_post, n_pre)
        ltd = self.config.A_minus * self.trace_post.t() @ s_pre  # (n_post, n_pre)

        dw = ltp - ltd

        # Mise à jour des poids (en place, avec clipping)
        with torch.no_grad():
            self.weight.data += self.config.lr_weights * dw
            self.weight.data.clamp_(-5.0, 5.0)

        return {'dw': dw, 'ltp': ltp, 'ltd': ltd}

    def reset_state(self) -> None:
        """Réinitialise les traces éligibles (début de séquence)."""
        device = self.trace_pre.device
        self.trace_pre = torch.zeros(1, self.n_pre, device=device)
        self.trace_post = torch.zeros(1, self.n_post, device=device)
        if self.delay_buffer is not None:
            self.delay_buffer.zero_()
        self._delay_idx = 0
