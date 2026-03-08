"""
experiments/toy_language.py — Expérience minimale sur des séquences linguistiques.

Dataset : séquences sujet-verbe-objet en français simplifié.
Structure : sujet → verbe → objet

Vocabulaire (~50 tokens) :
- Déterminants : le, la, les, un, une, des
- Sujets : chat, chien, oiseau, poisson, lapin, enfant, femme, homme
- Verbes : mange, boit, voit, aime, suit, entend, touche, cherche
- Objets : souris, eau, arbre, maison, fleur, balle, livre, pain

Pipeline :
1. Encoder le sujet dans Wernicke (représentation sémantique)
2. Transmettre via fascicule arqué en spikes
3. Broca génère verbe + objet en minimisant ε_B
4. Mesurer les métriques de performance

Métriques :
- Token accuracy : précision de génération
- Énergie libre F : convergence de l'inférence
- Paramètre d'ordre r : synchronisation de phase
- État Broca : CONVERGING/AMBIGUOUS/DIVERGING
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import SNNConfig
from core import OscillatoryClock
from modules import WernickeModule, BrocaModule, ArcuateFasciculus
from training.loss import variational_free_energy, FreeEnergyLoss
from graph.phase_sync import compute_order_parameter


# ── Vocabulaire ──────────────────────────────────────────────────────────────

VOCAB = [
    '<PAD>', '<BOS>', '<EOS>',
    # Déterminants
    'le', 'la', 'les', 'un', 'une',
    # Sujets (animés)
    'chat', 'chien', 'oiseau', 'poisson', 'lapin', 'enfant', 'femme', 'homme',
    # Verbes de base
    'mange', 'boit', 'voit', 'aime', 'suit', 'entend', 'touche', 'cherche',
    # Objets
    'souris', 'eau', 'arbre', 'maison', 'fleur', 'balle', 'livre', 'pain',
    'herbe', 'ciel', 'nuage', 'soleil', 'lune', 'étoile', 'mer', 'lac',
]
VOCAB_SIZE = len(VOCAB)
word2idx = {w: i for i, w in enumerate(VOCAB)}
idx2word = {i: w for i, w in enumerate(VOCAB)}

# ── Templates de phrases (sujet, verbe, objet) ───────────────────────────────

SENTENCE_TEMPLATES = [
    # (déterminant_sujet, sujet, verbe, déterminant_objet, objet)
    ('le', 'chat', 'mange', 'la', 'souris'),
    ('le', 'chien', 'boit', "l'", 'eau'),
    ('le', 'chat', 'voit', "l'", 'oiseau'),
    ("l'", 'oiseau', 'cherche', 'la', 'fleur'),
    ('le', 'lapin', 'mange', "l'", 'herbe'),
    ('le', 'chien', 'suit', 'le', 'chat'),
    ("l'", 'enfant', 'aime', 'le', 'lapin'),
    ("l'", 'enfant', 'cherche', 'la', 'balle'),
    ('la', 'femme', 'lit', 'le', 'livre'),
    ("l'", 'homme', 'voit', 'la', 'maison'),
    ('le', 'poisson', 'cherche', "l'", 'eau'),
    ('le', 'chat', 'entend', 'la', 'souris'),
]


def build_dataset(n_samples: int = 200) -> list[dict]:
    """
    Construit un dataset de séquences sujet-verbe-objet.

    Chaque sample est un dictionnaire avec :
    - 'subject'     : (vocab_size,) — one-hot du sujet
    - 'verb_idx'    : int — index du verbe attendu
    - 'object_idx'  : int — index de l'objet attendu
    - 'template'    : tuple — template source
    - 'sentence'    : str — phrase complète
    """
    dataset = []
    templates = [t for t in SENTENCE_TEMPLATES if all(w in word2idx for w in [t[1], t[2], t[4]])]

    for i in range(n_samples):
        tmpl = templates[i % len(templates)]
        det_s, subj, verb, det_o, obj = tmpl

        # Encoder le sujet comme one-hot (si dans le vocabulaire)
        subj_idx = word2idx.get(subj, 0)
        verb_idx = word2idx.get(verb, 0)
        obj_idx = word2idx.get(obj, 0)

        # Vecteur one-hot pour le sujet
        subj_onehot = torch.zeros(VOCAB_SIZE)
        subj_onehot[subj_idx] = 1.0

        dataset.append({
            'subject': subj_onehot,
            'verb_idx': verb_idx,
            'object_idx': obj_idx,
            'template': tmpl,
            'sentence': f"{det_s} {subj} {verb} {det_o} {obj}",
        })

    return dataset


class SNNPCLanguageModel(nn.Module):
    """
    Modèle hybride SNN-PC pour la génération de séquences linguistiques.

    Workflow complet :
    1. Sujet → Wernicke (encodage sémantique)
    2. Wernicke → Fascicule arqué → Broca (transmission en spikes)
    3. Broca génère verbe + objet
    4. Feedback Broca → Wernicke (correction)
    """

    def __init__(self, config: SNNConfig):
        super().__init__()
        self.config = config

        self.wernicke = WernickeModule(VOCAB_SIZE, config)
        self.broca = BrocaModule(VOCAB_SIZE, config)
        self.arcuate = ArcuateFasciculus(config)

        self.clock = OscillatoryClock(config)

    def forward(
        self,
        subject_input: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Passe forward complète : sujet → prédiction de verbe + objet.

        Args:
            subject_input : (batch, vocab_size) — one-hot du sujet
            context_idx   : (batch,) — index du token de contexte (BOS au départ)

        Returns:
            dict avec logits, erreurs, spikes, état Broca
        """
        batch = subject_input.shape[0]

        # ── 1. Avancer l'horloge ──────────────────────────────────────────────
        clock_state = self.clock.step()

        # ── 2. Module Wernicke ────────────────────────────────────────────────
        mu_prior = torch.zeros(batch, self.config.dim_wernicke,
                               device=subject_input.device)
        wernicke_out = self.wernicke(subject_input, mu_prior, self.clock)

        # ── 3. Transmission via fascicule arqué (W→B) ─────────────────────────
        message_W2B, reweight_W2B = self.arcuate.transmit(
            message=wernicke_out['prediction'],
            direction='W2B',
            visit_history=['W'],
        )

        # ── 4. Module Broca ───────────────────────────────────────────────────
        broca_out = self.broca(message_W2B, context_idx, self.clock)

        # ── 5. Feedback Broca → Wernicke (correction) ────────────────────────
        # Projeter la représentation de Broca vers dim_arcuate
        broca_feedback = broca_out['mu'][:, :self.config.dim_arcuate]
        message_B2W, reweight_B2W = self.arcuate.transmit(
            message=broca_feedback,
            direction='B2W',
            visit_history=['W', 'B'],
        )

        return {
            'logits': broca_out['logits'],
            'epsilon_W': wernicke_out['epsilon'],
            'epsilon_B': broca_out['epsilon'],
            'spikes_W': wernicke_out['spikes'],
            'spikes_B': broca_out['spikes'],
            'broca_state': broca_out['state'],
            'phase_coherence': broca_out['phase_coherence'],
            'reweight_W2B': reweight_W2B,
            'reweight_B2W': reweight_B2W,
            'clock_state': clock_state,
        }

    def reset_state(self) -> None:
        """Réinitialise tous les modules."""
        self.wernicke.reset_state()
        self.broca.reset_state()
        self.arcuate.reset_state()
        self.clock.reset()


def run_toy_language_experiment(
    n_epochs: int = 30,
    n_steps_per_sample: int = 20,
    batch_size: int = 8,
    save_dir: str = '../results',
) -> dict:
    """
    Expérience principale : entraînement et évaluation du modèle SNN-PC
    sur le dataset de séquences sujet-verbe-objet.

    Args:
        n_epochs          : nombre d'époques d'entraînement
        n_steps_per_sample: nombre de pas de simulation par sample
        batch_size        : taille des mini-batches
        save_dir          : répertoire de sauvegarde des figures

    Returns:
        dict avec l'historique des métriques
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = SNNConfig()
    model = SNNPCLanguageModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FreeEnergyLoss()

    dataset = build_dataset(n_samples=200)

    # Historiques des métriques
    history = {
        'free_energy': [],
        'order_param': [],
        'token_accuracy': [],
        'broca_states': [],
        'spike_trains_W': [],  # raster plots (sous-échantillonnés)
        'spike_trains_B': [],
    }

    print("═" * 60)
    print("Expérience : SNN-PC sur séquences sujet-verbe-objet")
    print(f"Vocabulaire : {VOCAB_SIZE} tokens | Dataset : {len(dataset)} samples")
    print(f"Config : dim_W={config.dim_wernicke}, dim_B={config.dim_broca}")
    print("═" * 60)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_order = 0.0
        n_batches = 0

        # Mélanger le dataset
        np.random.shuffle(dataset)

        for batch_start in range(0, len(dataset), batch_size):
            batch_data = dataset[batch_start:batch_start + batch_size]
            if len(batch_data) == 0:
                continue

            actual_batch = len(batch_data)

            # Préparer les tenseurs de batch
            subjects = torch.stack([d['subject'] for d in batch_data]).to(config.device)
            verb_targets = torch.tensor([d['verb_idx'] for d in batch_data],
                                        dtype=torch.long).to(config.device)
            obj_targets = torch.tensor([d['object_idx'] for d in batch_data],
                                       dtype=torch.long).to(config.device)

            # Token de contexte initial (BOS)
            context = torch.full((actual_batch,), word2idx['<BOS>'],
                                 dtype=torch.long).to(config.device)

            model.reset_state()
            optimizer.zero_grad()

            # ── Boucle temporelle (τ_pred) ────────────────────────────────────
            total_loss = torch.tensor(0.0, device=config.device)
            batch_spikes_W = []
            batch_spikes_B = []
            batch_order_params = []

            for t in range(n_steps_per_sample):
                # Passe forward
                out = model(subjects, context)

                # Perte : énergie libre + cross-entropy sur le verbe
                free_energy = loss_fn(out['epsilon_W'], out['epsilon_B'])
                ce_verb = F.cross_entropy(out['logits'], verb_targets)
                loss_t = free_energy + 0.5 * ce_verb

                total_loss = total_loss + loss_t

                # Mise à jour du contexte avec la prédiction courante (enseignant forcé)
                with torch.no_grad():
                    pred_tokens = out['logits'].argmax(dim=-1)
                    context = verb_targets  # forçage de l'enseignant

                # Enregistrer les trains de spikes (premier batch uniquement)
                if batch_start == 0:
                    batch_spikes_W.append(out['spikes_W'][0].detach().cpu().numpy()[:16])
                    batch_spikes_B.append(out['spikes_B'][0].detach().cpu().numpy()[:16])

                # Paramètre d'ordre (phases des neurones de Broca)
                phases = torch.rand(config.dim_broca) * 2 * 3.14159  # approximé
                order = compute_order_parameter(phases)
                batch_order_params.append(order)

            # Rétropropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Précision de génération (sur le verbe)
            with torch.no_grad():
                final_out = model(subjects, context)
                preds = final_out['logits'].argmax(dim=-1)
                acc = (preds == verb_targets).float().mean().item()

            epoch_loss += total_loss.item() / n_steps_per_sample
            epoch_acc += acc
            epoch_order += float(np.mean(batch_order_params))
            n_batches += 1

            # Enregistrer les trains de spikes du premier batch
            if batch_start == 0 and epoch == 0:
                history['spike_trains_W'] = batch_spikes_W
                history['spike_trains_B'] = batch_spikes_B

        # Métriques par époque
        avg_loss = epoch_loss / max(1, n_batches)
        avg_acc = epoch_acc / max(1, n_batches)
        avg_order = epoch_order / max(1, n_batches)

        history['free_energy'].append(avg_loss)
        history['token_accuracy'].append(avg_acc)
        history['order_param'].append(avg_order)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Époque {epoch+1:3d}/{n_epochs} | F={avg_loss:.4f} | "
                  f"Acc={avg_acc:.3f} | r={avg_order:.3f}")

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\nGénération des visualisations...")
    _plot_results(history, save_path)

    print(f"\nFigures sauvegardées dans : {save_path}")
    print(f"Précision finale : {history['token_accuracy'][-1]:.3f}")
    print(f"Énergie libre finale : {history['free_energy'][-1]:.4f}")

    return history


def _plot_results(history: dict, save_path: Path) -> None:
    """Génère et sauvegarde les 4 figures de l'expérience."""

    # ── Figure 1 : Raster plot des trains de spikes ───────────────────────────
    if history['spike_trains_W'] and history['spike_trains_B']:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        spikes_W = np.array(history['spike_trains_W'])  # (T, n_neurons)
        spikes_B = np.array(history['spike_trains_B'])

        T, n_W = spikes_W.shape
        T, n_B = spikes_B.shape

        # Raster plot Wernicke
        for neuron_idx in range(n_W):
            spike_times = np.where(spikes_W[:, neuron_idx] > 0.5)[0]
            axes[0].scatter(spike_times, np.full_like(spike_times, neuron_idx),
                           s=3, c='navy', alpha=0.7, marker='|')
        axes[0].set_ylabel('Neurone (Wernicke)')
        axes[0].set_title('Raster plot — Module Wernicke (sémantique)')
        axes[0].set_ylim(-0.5, n_W - 0.5)

        # Raster plot Broca
        for neuron_idx in range(n_B):
            spike_times = np.where(spikes_B[:, neuron_idx] > 0.5)[0]
            axes[1].scatter(spike_times, np.full_like(spike_times, neuron_idx),
                           s=3, c='darkred', alpha=0.7, marker='|')
        axes[1].set_ylabel('Neurone (Broca)')
        axes[1].set_title('Raster plot — Module Broca (syntaxique)')
        axes[1].set_ylim(-0.5, n_B - 0.5)
        axes[1].set_xlabel('Pas de temps (dt = 0.1ms)')

        plt.tight_layout()
        plt.savefig(save_path / 'raster_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 2 : Courbe F(t) pendant l'inférence ───────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 4))
    epochs = list(range(1, len(history['free_energy']) + 1))

    color_F = 'steelblue'
    ax1.plot(epochs, history['free_energy'], color=color_F, linewidth=2, label='F (énergie libre)')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Énergie libre F', color=color_F)
    ax1.tick_params(axis='y', labelcolor=color_F)
    ax1.set_title('Évolution de l\'énergie libre F et du paramètre d\'ordre r')

    ax2 = ax1.twinx()
    color_r = 'darkorange'
    ax2.plot(epochs, history['order_param'], color=color_r, linewidth=2,
             linestyle='--', label='r (Kuramoto)')
    ax2.set_ylabel('Paramètre d\'ordre r', color=color_r)
    ax2.tick_params(axis='y', labelcolor=color_r)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path / 'free_energy_kuramoto.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3 : Précision de génération ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history['token_accuracy'], color='seagreen', linewidth=2)
    ax.fill_between(epochs, history['token_accuracy'], alpha=0.2, color='seagreen')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Token Accuracy (verbe)')
    ax.set_title('Précision de génération — Module Broca')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0 / VOCAB_SIZE, color='gray', linestyle=':', label='Chance level')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'token_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ raster_plot.png")
    print(f"  ✓ free_energy_kuramoto.png")
    print(f"  ✓ token_accuracy.png")


if __name__ == '__main__':
    history = run_toy_language_experiment(
        n_epochs=30,
        n_steps_per_sample=20,
        batch_size=8,
        save_dir='../results',
    )
