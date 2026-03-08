"""
modules/wernicke.py — Module sémantique inspiré de l'aire de Wernicke.

L'aire de Wernicke (gyrus temporal supérieur gauche) est responsable de la
compréhension du langage — elle maintient une représentation sémantique de
ce qui est perçu (sons, mots, concepts).

Dans ce framework, Wernicke est un encodeur qui :
1. Reçoit l'input sensoriel (embedding de token)
2. Maintient la représentation sémantique μ_W(t) via inférence PC
3. Calcule l'erreur sémantique : ε_W = x_W - f(μ_W)
4. Envoie μ_W vers Broca via le fascicule arqué (encodé en spikes)

Dynamique d'inférence (mise à jour de μ_W) :
    dμ_W/dt = -ε_W + W^T · ε_higher
où ε_higher est l'erreur provenant du niveau hiérarchique supérieur (si existant).

En discrétisant avec le pas d'inférence η_pc :
    μ_W(t+1) = μ_W(t) + η_pc · (-ε_W(t) + top_down_correction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SNNConfig
from core.neuron import LIFNeuron
from core.oscillator import OscillatoryClock
from core.encoding import rate_encode


class WernickeModule(nn.Module):
    """
    Module encodeur sémantique (aire de Wernicke).

    Architecture interne :
    - Projection sensorielle : input → dim_wernicke
    - Couche de représentation LIF : maintient μ_W comme potentiel membranaire
    - Générateur de prédiction : μ_W → dim_arcuate (pour transmission vers Broca)

    La représentation sémantique μ_W est une variable d'état continue qui
    est mise à jour à chaque pas d'inférence pour minimiser ε_W.
    """

    def __init__(self, vocab_size: int, config: SNNConfig):
        """
        Args:
            vocab_size : taille du vocabulaire (dimension de l'input tokenisé)
            config     : configuration globale
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # ── Projection sensorielle ────────────────────────────────────────────
        # Transforme l'embedding de token en représentation sémantique initiale
        self.input_projection = nn.Linear(vocab_size, config.dim_wernicke)

        # ── Fonction de génération f(μ_W) ─────────────────────────────────────
        # f est la fonction qui prédit l'input observable depuis la représentation
        # Ici : couche linéaire + activation non-linéaire (tanh)
        self.prediction_fn = nn.Sequential(
            nn.Linear(config.dim_wernicke, config.dim_wernicke),
            nn.Tanh(),
        )

        # ── Générateur de message vers Broca ──────────────────────────────────
        # Projette μ_W dans l'espace du fascicule arqué (dim_arcuate)
        self.arcuate_projection = nn.Linear(config.dim_wernicke, config.dim_arcuate)

        # ── Couche LIF pour la dynamique temporelle ───────────────────────────
        # Les neurones LIF intègrent les erreurs ε_W et génèrent des spikes
        self.lif_neurons = LIFNeuron(config.dim_wernicke, config)

        # ── État interne : représentation sémantique μ_W ──────────────────────
        # Initialisé à zéro, mis à jour par inférence itérative
        self.register_buffer('mu_W', torch.zeros(1, config.dim_wernicke))

        # ── Prior μ_W,0 (prior sur la représentation sémantique) ──────────────
        # En l'absence d'input, μ_W est attiré vers ce prior par le terme -ε_W
        self.register_buffer('mu_prior', torch.zeros(1, config.dim_wernicke))

    def forward(
        self,
        x_input: torch.Tensor,
        mu_prior: torch.Tensor,
        clock: OscillatoryClock,
    ) -> dict[str, torch.Tensor]:
        """
        Un pas de traitement du module Wernicke.

        Étapes :
        1. Projection de l'input en représentation sémantique cible x_W
        2. Calcul de la prédiction f(μ_W)
        3. Calcul de l'erreur ε_W = x_W - f(μ_W)
        4. Mise à jour itérative de μ_W (n_inference_steps pas)
        5. Génération des spikes via LIF + encodage de phase
        6. Projection vers le fascicule arqué

        Args:
            x_input   : (batch, vocab_size) — représentation de l'input sensoriel
                        (peut être un one-hot, un embedding, ou des features audio)
            mu_prior  : (batch, dim_wernicke) — prior top-down (depuis un niveau supérieur)
            clock     : horloge oscillatoire pour le timing des spikes

        Returns:
            dict avec :
            - 'mu'        : (batch, dim_wernicke) — représentation sémantique
            - 'epsilon'   : (batch, dim_wernicke) — erreur de prédiction
            - 'spikes'    : (batch, dim_wernicke) — train de spikes
            - 'prediction': (batch, dim_arcuate)  — message vers Broca
        """
        batch = x_input.shape[0]
        target_device = x_input.device

        if self.mu_W.device != target_device:
            self.mu_W = self.mu_W.to(target_device)
        if self.mu_prior.device != target_device:
            self.mu_prior = self.mu_prior.to(target_device)
        if mu_prior.device != target_device:
            mu_prior = mu_prior.to(target_device)

        # ── 1. Projection de l'input sensoriel ───────────────────────────────
        # x_W = W_in · x_input + b_in : observation projetée dans l'espace sémantique
        x_W = self.input_projection(x_input)  # (batch, dim_wernicke)

        # ── 2. Inférence itérative de μ_W ─────────────────────────────────────
        # On part de l'état courant μ_W et on itère pour minimiser ε_W
        mu = self.mu_W.expand(batch, -1).clone()  # (batch, dim_wernicke)

        for _ in range(self.config.n_inference_steps):
            # Prédiction de l'observation depuis la représentation
            x_pred = self.prediction_fn(mu)  # (batch, dim_wernicke)

            # Erreur de prédiction sémantique
            epsilon = x_W - x_pred  # (batch, dim_wernicke)

            # Correction top-down depuis le prior (ou depuis Broca en boucle fermée)
            # mu_prior module la représentation selon les attentes contextuelles
            top_down = mu_prior - mu  # direction vers le prior

            # Règle de mise à jour PC :
            # dμ/dt = η_pc · (ε_W + prior_correction)
            # On utilise aussi la précision (1/σ²) comme poids de l'erreur
            mu = mu + self.config.eta_pc * (
                epsilon + self.config.precision_prior * top_down
            )

        # Erreur finale après convergence de l'inférence
        x_pred_final = self.prediction_fn(mu)
        epsilon_final = x_W - x_pred_final  # (batch, dim_wernicke)

        # ── 3. Mise à jour de l'état interne μ_W ─────────────────────────────
        self.mu_W = mu[0:1].detach()  # conserver pour le pas suivant

        # ── 4. Génération des spikes via couche LIF ───────────────────────────
        # Le courant d'entrée est la norme de μ (activité sémantique)
        # L'erreur PC module la dépolarisation membranaire
        I_syn = torch.tanh(mu)  # (batch, dim_wernicke) — courant normalisé
        gamma_phase = clock.get_gamma_phase()
        spikes, V_membrane = self.lif_neurons(I_syn, epsilon_final, gamma_phase)

        # ── 5. Projection vers le fascicule arqué ────────────────────────────
        # On projette μ_W (représentation continue) pour transmission à Broca
        # Note : on utilise μ (continu) plutôt que spikes (binaire) pour
        # préserver l'information sémantique lors de la transmission
        prediction_to_broca = self.arcuate_projection(mu)  # (batch, dim_arcuate)

        return {
            'mu': mu,
            'epsilon': epsilon_final,
            'spikes': spikes,
            'prediction': prediction_to_broca,
            'V_membrane': V_membrane,
        }

    def reset_state(self) -> None:
        """Réinitialise la représentation sémantique et les neurones LIF."""
        device = next(self.parameters()).device
        self.mu_W = torch.zeros(1, self.config.dim_wernicke, device=device)
        self.lif_neurons.reset_state()

    # ── Curriculum : contrôle du gel des paramètres ───────────────────────────

    def freeze(self) -> None:
        """
        Gèle tous les paramètres de Wernicke (utilisé en phases bootstrap et 1).
        Les buffers d'état (mu_W, etc.) ne sont pas affectés.
        """
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self, lr_scale: float = 1.0) -> dict:
        """
        Dégèle les paramètres avec un scaling optionnel du learning rate.

        Args:
            lr_scale : facteur multiplicatif sur le LR de base (ex: 0.1 pour
                       un dégel progressif en phase wikipedia_long).

        Returns:
            param_group : dict prêt à être passé à un optimizer Adam
                          {'params': [...], 'lr_scale': lr_scale}
        """
        for p in self.parameters():
            p.requires_grad_(True)
        return {'params': list(self.parameters()), 'lr_scale': lr_scale}

    def is_frozen(self) -> bool:
        """Retourne True si tous les paramètres sont gelés."""
        return not any(p.requires_grad for p in self.parameters())
