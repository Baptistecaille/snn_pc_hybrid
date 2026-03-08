"""
core/neuron.py — Neurone LIF (Leaky Integrate-and-Fire) augmenté par le Predictive Coding.

Le neurone LIF standard est étendu pour intégrer l'erreur de prédiction PC
directement dans sa dynamique membranaire :

    τ_m · dV/dt = -(V - V_rest) + R·I_syn(t) + η·ε(t)

où :
- τ_m : constante de temps membranaire (intégration des courants synaptiques)
- V   : potentiel membranaire
- V_rest : potentiel de repos
- R   : résistance membranaire (R = τ_m / C_m)
- I_syn : courant synaptique (décroissance exponentielle de τ_syn)
- η   : poids de l'erreur PC (eta_pc dans la config)
- ε(t) = r_observed(t) - r_predicted(t) : erreur de prédiction

La discrétisation utilise la méthode d'Euler explicite :
    V(t+dt) = V(t) + dt/τ_m · [-(V(t)-V_rest) + R·I_syn(t) + η·ε(t)]

Justification : Euler explicite est suffisant pour dt << τ_m (ici dt=0.1ms, τ_m=20ms).
"""

import torch
import torch.nn as nn
from config import SNNConfig
from training.surrogate import heaviside_surrogate


class LIFNeuron(nn.Module):
    """
    Neurone LIF augmenté par les erreurs de prédiction du Predictive Coding.

    Attributs d'état (enregistrés comme buffers PyTorch) :
    - V       : potentiel membranaire (batch, n_neurons)
    - I_syn   : courant synaptique interne (batch, n_neurons)
    - spike_history : historique des spikes pour le calcul STDP et du taux
    - t_last_spike  : timestamp du dernier spike par neurone (pour STDP)
    """

    def __init__(self, n_neurons: int, config: SNNConfig, beta: float = 1.0):
        """
        Args:
            n_neurons : nombre de neurones dans la couche
            config    : configuration globale SNNConfig
            beta      : paramètre du surrogate gradient (largeur de la fenêtre)
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.beta = beta

        # Paramètres dérivés (scalaires, non entraînables)
        self.alpha_m = 1.0 - config.dt / config.tau_m      # facteur de fuite membranaire
        self.alpha_syn = 1.0 - config.dt / config.tau_syn  # facteur de décroissance synaptique

        # États internes — initialisés comme buffers (persistent mais non-paramètres)
        # Forme : (1, n_neurons) pour broadcast avec des batches arbitraires
        self.register_buffer('V', torch.full((1, n_neurons), config.v_rest))
        self.register_buffer('I_syn_state', torch.zeros(1, n_neurons))

        # Historique des spikes sur une fenêtre glissante (pour calcul du taux)
        # Taille : steps_per_rate_window pas de simulation
        self.spike_history: list[torch.Tensor] = []
        self.max_history_steps = 500  # ≈ 50ms à dt=0.1ms

        # Timestamp du dernier spike (pour la règle STDP dans synapse.py)
        self.register_buffer('t_last_spike', torch.full((1, n_neurons), -1e6))
        self.current_time: float = 0.0

    def forward(
        self,
        I_syn: torch.Tensor,
        epsilon: torch.Tensor,
        phase: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Un pas de simulation dt du neurone LIF augmenté PC.

        Équation discrète :
            I_syn_state(t+1) = α_syn · I_syn_state(t) + I_syn(t)
            V(t+1) = α_m · V(t) + (1-α_m)·V_rest + R·I_total·dt/τ_m + η·ε(t)·dt/τ_m
            spike   = H(V(t+1) - V_threshold)
            V(t+1) = spike·V_reset + (1-spike)·V(t+1)   [reset post-spike]

        Note : l'erreur PC ε(t) agit comme un courant additionnel modulant
        la dépolarisation. Un ε positif (observation > prédiction) dépolarise
        le neurone, l'amenant plus rapidement au seuil.

        Args:
            I_syn   : (batch, n_neurons) — courant synaptique entrant à ce pas
            epsilon : (batch, n_neurons) — erreur de prédiction PC
            phase   : phase gamma courante ∈ [0, 2π] (pour modulation future)

        Returns:
            spikes     : (batch, n_neurons) ∈ {0.0, 1.0} — via surrogate gradient
            V_membrane : (batch, n_neurons) — potentiel membranaire après mise à jour
        """
        batch = I_syn.shape[0]

        # Expansion de l'état interne pour le batch courant
        V = self.V.expand(batch, -1).clone()
        I_syn_state = self.I_syn_state.expand(batch, -1).clone()

        # ── 1. Mise à jour du courant synaptique (filtre passe-bas d'ordre 1) ──
        # Discrétisation : I(t+dt) = (1 - dt/τ_syn)·I(t) + I_entrant(t)
        I_syn_state = self.alpha_syn * I_syn_state + I_syn

        # ── 2. Courant total intégré au potentiel membranaire ──
        # Facteur dt/τ_m = (1 - α_m) pour la mise à l'échelle
        R = self.config.R_membrane
        dt_over_tau = self.config.dt / self.config.tau_m

        # Terme synaptique + terme d'erreur PC
        I_total = R * I_syn_state + self.config.eta_pc * epsilon

        # Mise à jour du potentiel (Euler explicite)
        # V(t+dt) = α_m·V + (1-α_m)·V_rest + dt/τ_m · I_total
        V_new = self.alpha_m * V + (1.0 - self.alpha_m) * self.config.v_rest + dt_over_tau * I_total

        # ── 3. Génération des spikes via surrogate gradient ──
        # x = V - V_threshold : positif → spike
        x = V_new - self.config.v_threshold
        spikes = heaviside_surrogate(x, self.beta)

        # ── 4. Reset membranaire post-spike ──
        # V_reset si spike, sinon V_new (opération différentiable car spikes ∈ {0,1})
        V_new = spikes * self.config.v_reset + (1.0 - spikes) * V_new

        # ── 5. Mise à jour des états internes (premier élément du batch comme référence) ──
        # Note : on stocke le premier batch pour la continuité temporelle.
        # En production, on passerait l'état complet batch comme buffer.
        self.V = V_new[0:1].detach()
        self.I_syn_state = I_syn_state[0:1].detach()

        # Mise à jour de l'historique des spikes et du temps
        self.spike_history.append(spikes.detach())
        if len(self.spike_history) > self.max_history_steps:
            self.spike_history.pop(0)

        # Mise à jour des timestamps de dernier spike (pour STDP)
        fired = (spikes[0:1] > 0.5).float()
        self.t_last_spike = fired * self.current_time + (1.0 - fired) * self.t_last_spike
        self.current_time += self.config.dt

        return spikes, V_new

    def reset_state(self, batch_size: int = 1) -> None:
        """
        Réinitialise tous les états internes au début d'une nouvelle séquence.

        Args:
            batch_size : taille du batch (pour l'initialisation correcte des buffers)
        """
        device = self.V.device
        self.V = torch.full((1, self.n_neurons), self.config.v_rest, device=device)
        self.I_syn_state = torch.zeros(1, self.n_neurons, device=device)
        self.spike_history = []
        self.t_last_spike = torch.full((1, self.n_neurons), -1e6, device=device)
        self.current_time = 0.0

    def get_firing_rate(self, window_ms: float = 50.0) -> torch.Tensor:
        """
        Calcule le taux de décharge moyen sur une fenêtre temporelle glissante.

        Args:
            window_ms : durée de la fenêtre en millisecondes

        Returns:
            rates : (n_neurons,) — taux de décharge en spikes/ms
        """
        n_steps = max(1, int(window_ms / self.config.dt))
        n_steps = min(n_steps, len(self.spike_history))

        if n_steps == 0:
            device = self.V.device
            return torch.zeros(self.n_neurons, device=device)

        recent_spikes = torch.stack(self.spike_history[-n_steps:], dim=0)  # (T, batch, n)
        # Moyenne sur le temps et le batch (dim 0 et 1)
        rates = recent_spikes.float().mean(dim=(0, 1))  # (n_neurons,)
        return rates / self.config.dt  # convertir en spikes/ms
