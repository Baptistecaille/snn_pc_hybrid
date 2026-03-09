"""
modules/broca.py — Module syntaxique inspiré de l'aire de Broca.

L'aire de Broca (gyrus frontal inférieur gauche, aire 44/45) est responsable
de la production du langage et du traitement syntaxique. Elle reçoit des
informations sémantiques de Wernicke et les convertit en séquences de tokens.

Dans ce framework, Broca est un décodeur qui :
1. Reçoit μ_W depuis le fascicule arqué (représentation sémantique en spikes)
2. Maintient un contexte syntaxique x_B
3. Calcule l'erreur de production : ε_B = x_B - f(μ_W)
4. Génère des tokens en minimisant ε_B via inférence PC
5. Surveille l'état de convergence pour détecter les ambiguïtés/contradictions

États de convergence :
- CONVERGING : ε_B → 0, génération fluide (accord sémantique-syntaxique)
- AMBIGUOUS  : ε_B oscille autour d'un minimum non-nul (ambiguïté structurelle)
- DIVERGING  : ε_B croît, contradiction entre sémantique et syntaxe attendue

Référence : Hagoort (2005) "On Broca, brain, and binding: a new framework"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SNNConfig
from core.neuron import LIFNeuron
from core.oscillator import OscillatoryClock


# Seuils pour la classification de l'état de convergence
_EPSILON_CONVERGING = 0.1   # ε_B en dessous → CONVERGING
_EPSILON_DIVERGING = 1.0    # ε_B au dessus → DIVERGING
_OSCILLATION_WINDOW = 5     # nombre de pas pour détecter les oscillations


class BrocaModule(nn.Module):
    """
    Module décodeur syntaxique (aire de Broca).

    Architecture interne :
    - Réception depuis fascicule arqué : dim_arcuate → dim_broca
    - Couche contextuelle : contexte syntaxique (tokens précédents)
    - Couche LIF : dynamique temporelle de la représentation syntaxique
    - Tête de génération : dim_broca → vocab_size (logits)
    - Moniteur d'état : détecte CONVERGING/AMBIGUOUS/DIVERGING

    La représentation syntaxique μ_B est mise à jour par inférence PC pour
    minimiser l'écart entre la prédiction sémantique (depuis Wernicke) et
    la contrainte syntaxique locale (contexte).
    """

    def __init__(self, vocab_size: int, config: SNNConfig):
        """
        Args:
            vocab_size : taille du vocabulaire (dimension des logits de sortie)
            config     : configuration globale
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # ── Réception depuis le fascicule arqué ───────────────────────────────
        # Projette le message de Wernicke dans l'espace syntaxique de Broca
        self.arcuate_reception = nn.Linear(config.dim_arcuate, config.dim_broca)

        # ── Contexte syntaxique (tokens précédents) ───────────────────────────
        # Embedding des tokens de contexte + projection dans dim_broca
        self.context_embedding = nn.Embedding(vocab_size, config.dim_broca)
        self.context_projection = nn.Linear(config.dim_broca, config.dim_broca)

        # ── Fonction de prédiction : μ_W → x_B prédit ────────────────────────
        # Prédit quelle représentation syntaxique devrait correspondre à μ_W
        self.prediction_fn = nn.Sequential(
            nn.Linear(config.dim_broca, config.dim_broca),
            nn.Tanh(),
            nn.Linear(config.dim_broca, config.dim_broca),
        )

        # ── Tête de génération ────────────────────────────────────────────────
        # Projette la représentation syntaxique vers le vocabulaire
        self.output_head = nn.Linear(config.dim_broca, vocab_size)

        # ── Couche LIF pour la dynamique spikante ────────────────────────────
        self.lif_neurons = LIFNeuron(config.dim_broca, config)

        # ── États internes ─────────────────────────────────────────────────────
        self.register_buffer('mu_B', torch.zeros(1, config.dim_broca))

        # Historique des erreurs pour la détection d'oscillations
        self._epsilon_history: list[float] = []

    def _ensure_state_shape(self, batch_size: int, device: torch.device) -> None:
        if self.mu_B.device != device:
            self.mu_B = self.mu_B.to(device)

        if self.mu_B.shape[0] != batch_size:
            self.mu_B = torch.zeros(batch_size, self.config.dim_broca, device=device)
            self._epsilon_history = []

    def forward(
        self,
        mu_wernicke: torch.Tensor,
        x_context: torch.Tensor,
        clock: OscillatoryClock,
    ) -> dict[str, torch.Tensor]:
        """
        Un pas de traitement du module Broca.

        Étapes :
        1. Réception du message sémantique depuis le fascicule arqué
        2. Calcul de la représentation syntaxique du contexte
        3. Calcul de l'erreur de production ε_B
        4. Inférence itérative de μ_B
        5. Génération des logits et des spikes
        6. Évaluation de l'état de convergence

        Args:
            mu_wernicke : (batch, dim_arcuate) — représentation sémantique depuis Wernicke
            x_context   : (batch,) ou (batch, dim_broca) — contexte syntaxique local
                          Si 1D : indices de tokens; si 2D : embeddings déjà projetés
            clock       : horloge oscillatoire

        Returns:
            dict avec :
            - 'logits'         : (batch, vocab_size) — distribution sur le vocabulaire
            - 'epsilon'        : (batch, dim_broca) — erreur de production
            - 'spikes'         : (batch, dim_broca) — train de spikes
            - 'state'          : str — état de convergence
            - 'phase_coherence': float — cohérence de phase ∈ [0, 1]
        """
        batch = mu_wernicke.shape[0]
        target_device = mu_wernicke.device

        self._ensure_state_shape(batch, target_device)
        if x_context.device != target_device:
            x_context = x_context.to(target_device)

        # ── 1. Projection du message sémantique ───────────────────────────────
        # Transforme le message de dim_arcuate en représentation dans dim_broca
        semantic_input = self.arcuate_reception(mu_wernicke)  # (batch, dim_broca)
        semantic_input = torch.tanh(semantic_input)

        # ── 2. Contexte syntaxique ────────────────────────────────────────────
        if x_context.dim() == 1:
            # x_context est un vecteur d'indices de tokens
            context_emb = self.context_embedding(x_context.long())  # (batch, dim_broca)
        else:
            # x_context est déjà une représentation continue
            context_emb = x_context

        x_B = self.context_projection(context_emb)  # (batch, dim_broca)
        x_B = torch.tanh(x_B)

        # ── 3. Inférence itérative de μ_B ─────────────────────────────────────
        # L'inférence minimise : F_B = ||ε_B||² = ||x_B - f(μ_B)||²
        # La mise à jour est : dμ_B/dt = η_pc · ε_B + correction sémantique
        mu = self.mu_B.clone()  # (batch, dim_broca)

        for step in range(self.config.n_inference_steps):
            # Prédiction de l'observation syntaxique depuis μ_B
            x_B_pred = self.prediction_fn(mu)  # (batch, dim_broca)

            # Erreur de production : écart entre contexte syntaxique et prédiction
            epsilon = x_B - x_B_pred  # (batch, dim_broca)

            # Correction sémantique : μ_B est attiré vers la représentation sémantique
            semantic_correction = semantic_input - mu

            # Règle de mise à jour PC pour Broca
            mu = mu + self.config.eta_pc * (
                epsilon + 0.5 * semantic_correction
            )

        # Erreur finale
        x_B_pred_final = self.prediction_fn(mu)
        epsilon_final = x_B - x_B_pred_final  # (batch, dim_broca)

        # ── 4. Mise à jour de l'état interne ──────────────────────────────────
        self.mu_B = mu.detach()

        # ── 5. Génération des logits ──────────────────────────────────────────
        logits = self.output_head(mu)  # (batch, vocab_size)

        # ── 6. Génération des spikes via couche LIF ───────────────────────────
        I_syn = torch.tanh(mu)
        gamma_phase = clock.get_gamma_phase()
        spikes, V_membrane = self.lif_neurons(I_syn, epsilon_final, gamma_phase)

        # ── 7. Évaluation de l'état de convergence ────────────────────────────
        epsilon_norm = epsilon_final.norm(dim=-1).mean().item()
        self._epsilon_history.append(epsilon_norm)
        if len(self._epsilon_history) > _OSCILLATION_WINDOW:
            self._epsilon_history.pop(0)

        state = self._classify_state(epsilon_norm)
        phase_coherence = self._compute_phase_coherence(spikes, clock)

        return {
            'logits': logits,
            'epsilon': epsilon_final,
            'spikes': spikes,
            'state': state,
            'phase_coherence': phase_coherence,
            'mu': mu,
            'V_membrane': V_membrane,
        }

    def _classify_state(self, epsilon_norm: float) -> str:
        """
        Classifie l'état de convergence de Broca selon l'historique des erreurs.

        Logique :
        - CONVERGING : ε décroît et reste < seuil_bas
        - DIVERGING  : ε croît et dépasse seuil_haut
        - AMBIGUOUS  : ε oscille (variance de l'historique élevée)

        Args:
            epsilon_norm : norme actuelle de l'erreur de production

        Returns:
            état : 'CONVERGING' | 'AMBIGUOUS' | 'DIVERGING'
        """
        if epsilon_norm < _EPSILON_CONVERGING:
            return 'CONVERGING'

        if epsilon_norm > _EPSILON_DIVERGING:
            return 'DIVERGING'

        # Vérifier les oscillations sur l'historique récent
        if len(self._epsilon_history) >= _OSCILLATION_WINDOW:
            import statistics
            variance = statistics.variance(self._epsilon_history)
            # Si la variance est élevée relative à la moyenne → oscillation
            mean_eps = statistics.mean(self._epsilon_history)
            if mean_eps > 0 and variance / (mean_eps ** 2) > 0.1:
                return 'AMBIGUOUS'

        return 'CONVERGING'

    def _compute_phase_coherence(
        self, spikes: torch.Tensor, clock: OscillatoryClock
    ) -> float:
        """
        Mesure de cohérence de phase entre les neurones actifs de Broca.

        Utilise le paramètre d'ordre de Kuramoto restreint aux neurones qui spikent.
        Ici, on approxime la cohérence par la fraction de neurones synchrones
        (proportion qui spikent au même pas gamma).

        Returns:
            cohérence ∈ [0, 1] — 1 = synchronisation parfaite, 0 = désynchronisation
        """
        n_spikes = spikes.float().sum().item()
        n_total = spikes.numel()
        if n_total == 0:
            return 0.0
        # Proportion de neurones actifs × amplitude gamma (modulation thêta-gamma)
        active_fraction = n_spikes / n_total
        gamma_amp = clock.get_gamma_amplitude()
        return float(active_fraction * gamma_amp)

    def reset_state(self, batch_size: int = 1) -> None:
        """Réinitialise la représentation syntaxique et les neurones LIF."""
        device = next(self.parameters()).device
        self.mu_B = torch.zeros(batch_size, self.config.dim_broca, device=device)
        self._epsilon_history = []
        self.lif_neurons.reset_state(batch_size=batch_size)
        self._epsilon_history = []

    # ── Curriculum : contrôle du gel des paramètres ───────────────────────────

    def freeze(self) -> None:
        """
        Gèle tous les paramètres de Broca.
        Utilisé en phases 'bootstrap' et 'wikipedia_short' pour forcer
        Wernicke à apprendre en premier.
        """
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self, lr_scale: float = 1.0) -> dict:
        """
        Dégèle les paramètres de Broca avec un scaling du LR.

        Args:
            lr_scale : 0.1 en phase 'wikipedia_long' (dégel progressif),
                       1.0 en phases 'oscar_*' (entraînement complet).

        Returns:
            param_group : dict pour l'optimizer Adam
        """
        for p in self.parameters():
            p.requires_grad_(True)
        return {'params': list(self.parameters()), 'lr_scale': lr_scale}

    def is_frozen(self) -> bool:
        """Retourne True si tous les paramètres sont gelés."""
        return not any(p.requires_grad for p in self.parameters())
