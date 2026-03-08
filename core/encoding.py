"""
core/encoding.py — Encodage des erreurs de prédiction en trains de spikes.

Deux schémas d'encodage sont implémentés, chacun avec des propriétés théoriques
différentes :

1. Rate coding : l'erreur ε est encodée comme un taux de décharge (probabilité
   de spike par pas de temps). Simple mais perd l'information temporelle fine.

2. Phase coding (temporal coding) : les neurones avec une grande erreur spikent
   tôt dans la phase gamma, ceux avec une petite erreur spikent tard ou pas du tout.
   Permet d'encoder une information continue dans le timing sub-ms.

Référence théorique :
- Panzeri et al. (2010) "Sensory neural codes using multiplexed temporal scales"
- Brette (2015) "Philosophy of the Spike: Rate-Based vs. Spike-Based Theories of the Brain"
"""

import math
import torch


def rate_encode(
    epsilon: torch.Tensor,
    dt: float,
    max_rate: float = 1.0,
) -> torch.Tensor:
    """
    Encodage par taux : convertit une erreur de prédiction en train de spikes.

    La probabilité de spike à chaque pas de temps est :
        p(spike) = sigmoid(ε) · dt · max_rate

    sigmoid(ε) ∈ (0, 1) mappe l'erreur (qui peut être négative) vers
    une probabilité valide. max_rate normalise pour que le taux maximal
    soit en spikes/ms.

    Args:
        epsilon  : (batch, n_neurons) — erreur de prédiction (peut être négative)
        dt       : pas temporel en ms
        max_rate : taux de décharge maximal en spikes/ms (défaut = 1 spike/ms)

    Returns:
        spikes : (batch, n_neurons) ∈ {0, 1} — train de spikes stochastique

    Note : torch.bernoulli() est non-différentiable. Pour l'apprentissage,
    utiliser les spikes de LIFNeuron (qui utilisent le surrogate gradient).
    Cette fonction est utilisée pour l'initialisation et l'encodage sensoriel.
    """
    # Taux de décharge normalisé ∈ (0, max_rate · dt)
    rate = torch.sigmoid(epsilon) * max_rate * dt
    # Clamp pour garantir une probabilité valide malgré les erreurs numériques
    rate = rate.clamp(0.0, 1.0)
    return torch.bernoulli(rate)


def phase_encode(
    epsilon: torch.Tensor,
    phase: float,
    epsilon_max: float = 5.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Encodage de phase : les neurones avec grande erreur spikent tôt dans le cycle gamma.

    Principe : le timing du spike encode la magnitude de l'erreur.
    Le spike survient au temps :
        t_spike = t_ref + α · (1 - ε_norm)
    où ε_norm = clamp(|ε| / ε_max, 0, 1) est l'erreur normalisée.

    Un neurone spike si la phase courante φ correspond au timing prédit :
        spike = 1 si φ ≈ 2π · (1 - ε_norm)  (modulo un seuil de tolérance)

    Intuition : ε_norm = 1 (grande erreur) → t_spike = t_ref (spike tôt, phase ≈ 0)
                ε_norm = 0 (petite erreur) → t_spike = t_ref + α (spike tard, phase ≈ 2π)

    Args:
        epsilon     : (batch, n_neurons) — erreur de prédiction
        phase       : phase gamma courante ∈ [0, 2π] (fournie par OscillatoryClock)
        epsilon_max : valeur maximale d'erreur pour la normalisation
        alpha       : largeur de la fenêtre de phase (en radians)

    Returns:
        spikes : (batch, n_neurons) ∈ {0.0, 1.0} — masque binaire de phase
    """
    # Normaliser l'erreur dans [0, 1] (on prend la valeur absolue)
    epsilon_norm = (epsilon.abs() / epsilon_max).clamp(0.0, 1.0)

    # Phase cible pour chaque neurone : grande erreur → phase cible proche de 0
    phase_target = 2.0 * math.pi * (1.0 - epsilon_norm)  # (batch, n_neurons) ∈ [0, 2π]

    # Tolérance angulaire : demi-largeur de la fenêtre
    # La tolérance est proportionnelle à α (1/2 radians par défaut)
    tolerance = alpha / 2.0

    # Distance angulaire entre phase courante et phase cible (modulo 2π)
    # Formule de distance circulaire : min(|Δφ|, 2π - |Δφ|)
    delta_phase = (phase - phase_target).abs()
    delta_phase = torch.where(delta_phase > math.pi, 2.0 * math.pi - delta_phase, delta_phase)

    # Spike si la phase courante est dans la fenêtre de tolérance
    spikes = (delta_phase < tolerance).float()

    return spikes


def population_rate_to_value(
    spikes: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    """
    Décode un train de spikes populationnel en une valeur continue.

    Utilisé pour reconstruire les représentations continues depuis les trains
    de spikes — opération inverse de rate_encode.

    Args:
        spikes       : (T, batch, n_neurons) — séquence de spikes
        window_steps : nombre de pas pour la moyenne glissante

    Returns:
        rates : (batch, n_neurons) — taux moyen estimé ∈ [0, 1]
    """
    T = spikes.shape[0]
    n_steps = min(window_steps, T)
    return spikes[-n_steps:].float().mean(dim=0)


def burst_encode(
    epsilon: torch.Tensor,
    dt: float,
    burst_threshold: float = 2.0,
    max_burst: int = 3,
) -> torch.Tensor:
    """
    Encodage par burst : les grandes erreurs génèrent des bursts de spikes.

    Un burst est une rafale rapide de N spikes (N proportionnel à |ε|).
    Ce schéma est biologiquement observé dans le cortex sensoriel lors de
    stimuli saillants.

    Args:
        epsilon         : (batch, n_neurons) — erreur de prédiction
        dt              : pas temporel en ms
        burst_threshold : seuil à partir duquel un burst est déclenché
        max_burst       : nombre maximal de spikes par burst

    Returns:
        spikes : (batch, n_neurons) — spikes (peut dépasser 1 pour les bursts)
    """
    abs_epsilon = epsilon.abs()

    # Nombre de spikes par neurone (0 si sous le seuil, jusqu'à max_burst)
    n_spikes = ((abs_epsilon - burst_threshold) / burst_threshold).clamp(0.0, 1.0)
    n_spikes = (n_spikes * max_burst).floor()

    # Convertir en probabilité pour le pas courant
    # On émet 1 spike si Bernoulli(n_spikes/max_burst) réussit
    p_spike = (n_spikes / max_burst).clamp(0.0, 1.0) * dt
    return torch.bernoulli(p_spike.clamp(0.0, 1.0))
