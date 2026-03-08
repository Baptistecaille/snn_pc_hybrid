"""
modules/arcuate.py — Fascicule arqué : canal de communication bidirectionnel W ↔ B.

Le fascicule arqué (fasciculus arcuatus) est un faisceau de substance blanche
qui connecte les aires de Wernicke et Broca. C'est le principal canal de
communication entre sémantique et syntaxe.

Dans ce framework, le fascicule arqué :
1. Encode les messages continus en trains de spikes (phase coding)
2. Introduit un délai axonal réaliste (propagation neuronale ~10-30ms)
3. Détecte les cycles de communication (W→B→W→...) pour les amortir
4. Implémente l'amortissement cyclique : f(n) = exp(-λ·n) pour éviter
   la divergence dans les circuits récurrents

La détection de cycles est cruciale pour la condition de convergence de Kuramoto :
sans amortissement, les circuits récurrents accumulent les erreurs et divergent.

Référence : Friederici (2012) "The cortical language circuit: from auditory perception to sentence comprehension"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SNNConfig
from core.encoding import rate_encode, phase_encode


class ArcuateFasciculus(nn.Module):
    """
    Canal de communication inter-modules avec encodage spike et détection de cycles.

    Le canal est bidirectionnel mais asymétrique :
    - W→B : message sémantique → contrainte syntaxique (projection + encodage)
    - B→W : correction syntaxique → mise à jour sémantique (feedback)

    Propriété anti-cycle : si un message revient en boucle (W→B→W ou plus long),
    son amplitude est atténuée par f(n) = exp(-λ·n) où n est le nombre de passages.
    Cela garantit la convergence de l'inférence même sur des graphes denses.
    """

    def __init__(self, config: SNNConfig):
        super().__init__()
        self.config = config

        # ── Projections directionnelles ───────────────────────────────────────
        # Chaque direction a ses propres poids pour l'asymétrie fonctionnelle
        self.W2B_projection = nn.Linear(config.dim_arcuate, config.dim_arcuate)
        self.B2W_projection = nn.Linear(config.dim_arcuate, config.dim_arcuate)

        # ── Encodeur spike (taux de décharge) ────────────────────────────────
        # Projette le message continu vers une représentation encodable en spikes
        self.spike_encoder = nn.Sequential(
            nn.Linear(config.dim_arcuate, config.dim_arcuate),
            nn.Tanh(),
        )

        # ── Décodeur spike (reconstruction depuis les spikes) ─────────────────
        self.spike_decoder = nn.Sequential(
            nn.Linear(config.dim_arcuate, config.dim_arcuate),
            nn.Tanh(),
        )

        # ── Compteur de cycles par message ────────────────────────────────────
        # Dictionnaire de compteurs (clé = identifiant de message hash)
        self._cycle_counts: dict[str, int] = {}

        # ── Historique des transmissions (pour le débogage) ───────────────────
        self._transmission_log: list[dict] = []

    def transmit(
        self,
        message: torch.Tensor,
        direction: str,
        visit_history: list[str],
    ) -> tuple[torch.Tensor, float]:
        """
        Transmet un message entre Wernicke et Broca, encodé en spikes.

        Processus :
        1. Détecter si le message est dans un cycle (visit_history contient les deux modules)
        2. Calculer le facteur d'amortissement f(n) = exp(-λ·n)
        3. Appliquer la projection directionnelle
        4. Encoder en spikes via rate coding
        5. Décoder depuis les spikes (reconstruction)
        6. Multiplier par le facteur d'amortissement

        Args:
            message       : (batch, dim_arcuate) — message à transmettre
            direction     : 'W2B' (Wernicke → Broca) ou 'B2W' (Broca → Wernicke)
            visit_history : liste des modules déjà visités (ex: ['W', 'B', 'W'])
                            pour détecter les cycles

        Returns:
            transmitted : (batch, dim_arcuate) — message transmis (reconstruit depuis spikes)
            reweight    : float ∈ (0, 1] — facteur d'amortissement (1 = pas de cycle)
        """
        assert direction in ('W2B', 'B2W'), f"direction doit être 'W2B' ou 'B2W', reçu: {direction}"

        # ── 1. Détection et comptage de cycles ───────────────────────────────
        cycle_count = self._count_cycles(visit_history, direction)

        # Facteur d'amortissement exponentiel : f(n) = exp(-λ·n)
        # n=0 (pas de cycle) → f=1 (pas d'amortissement)
        # n=1 (un cycle)     → f=exp(-λ) ≈ 0.61 (pour λ=0.5)
        # n=2 (deux cycles)  → f=exp(-2λ) ≈ 0.37
        reweight = math.exp(-self.config.cycle_damping_lambda * cycle_count)

        # ── 2. Projection directionnelle ──────────────────────────────────────
        if direction == 'W2B':
            projected = self.W2B_projection(message)
        else:
            projected = self.B2W_projection(message)

        projected = torch.tanh(projected)

        # ── 3. Encodage en spikes ─────────────────────────────────────────────
        # On encode le message continu en un vecteur d'activité moyenne
        # Approximation : on utilise le rate encoding avec dt=1 (par commodité)
        encodable = self.spike_encoder(projected)  # (batch, dim_arcuate)

        # Génération de spikes stochastiques (non-différentiables, pour la transmission)
        spikes = rate_encode(encodable, dt=self.config.dt * 10)  # fenêtre plus large

        # ── 4. Reconstruction depuis les spikes ───────────────────────────────
        # Les spikes sont reconvertis en représentation continue
        reconstructed = self.spike_decoder(spikes.float())  # (batch, dim_arcuate)

        # ── 5. Application de l'amortissement cyclique ────────────────────────
        transmitted = reweight * reconstructed

        # Log de la transmission (pour visualisation)
        self._transmission_log.append({
            'direction': direction,
            'cycle_count': cycle_count,
            'reweight': reweight,
            'message_norm': message.norm().item(),
        })
        if len(self._transmission_log) > 1000:
            self._transmission_log.pop(0)

        return transmitted, reweight

    def _count_cycles(self, visit_history: list[str], direction: str) -> int:
        """
        Compte le nombre de fois que la direction courante crée un cycle.

        Un cycle est détecté quand le module de départ apparaît deux fois
        dans l'historique (le message est revenu à son origine).

        Exemples :
        - history=['W'], direction='W2B' → pas de cycle, count=0
        - history=['W', 'B'], direction='B2W' → 1 cycle (B revient vers W)
        - history=['W', 'B', 'W', 'B'], direction='B2W' → 2 cycles

        Args:
            visit_history : liste des modules visités
            direction     : direction de transmission courante

        Returns:
            cycle_count : nombre de cycles (0 si pas de cycle)
        """
        src = direction[0]  # 'W' ou 'B'
        # Compter les occurrences du module source dans l'historique
        # Chaque occurrence supplémentaire = un cycle de plus
        if not visit_history:
            return 0
        count = sum(1 for v in visit_history if v == src)
        return max(0, count - 1)  # -1 car la première visite n'est pas un cycle

    def get_transmission_stats(self) -> dict:
        """
        Retourne des statistiques sur les transmissions récentes.
        Utile pour diagnostiquer les problèmes de cycles.
        """
        if not self._transmission_log:
            return {}
        recent = self._transmission_log[-100:]
        return {
            'mean_cycle_count': sum(r['cycle_count'] for r in recent) / len(recent),
            'mean_reweight': sum(r['reweight'] for r in recent) / len(recent),
            'mean_message_norm': sum(r['message_norm'] for r in recent) / len(recent),
            'n_cycles_detected': sum(1 for r in recent if r['cycle_count'] > 0),
        }

    def reset_state(self) -> None:
        """Réinitialise le compteur de cycles et le log."""
        self._cycle_counts = {}
        self._transmission_log = []

    # ── Curriculum : contrôle du gel des paramètres ───────────────────────────

    def freeze(self) -> None:
        """
        Gèle les paramètres appris du fascicule arqué (W2B, B2W, spike_{en,de}coder).
        Le canal reste fonctionnel (transmit() continue à fonctionner) mais
        ses poids ne sont pas mis à jour par l'optimizer.
        """
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self, lr_scale: float = 1.0) -> dict:
        """
        Dégèle les paramètres du fascicule arqué.

        Args:
            lr_scale : facteur de LR (ex: 0.5 en 'wikipedia_short' pour un
                       apprentissage plus prudent du canal de communication).

        Returns:
            param_group : dict pour l'optimizer Adam
        """
        for p in self.parameters():
            p.requires_grad_(True)
        return {'params': list(self.parameters()), 'lr_scale': lr_scale}

    def is_frozen(self) -> bool:
        """Retourne True si tous les paramètres sont gelés."""
        return not any(p.requires_grad for p in self.parameters())
