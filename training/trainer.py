"""
training/trainer.py — Trainer unifié avec curriculum en 5 phases.

Orchestre les modules existants (WernickeModule, BrocaModule, ArcuateFasciculus)
en respectant le curriculum défini dans SNNConfig :

    bootstrap → wikipedia_short → wikipedia_long → oscar_filtered → oscar_full

Stratégie de dégel par phase :
    'bootstrap'       : Wernicke seul (Broca + Arcuate gelés)
    'wikipedia_short' : Wernicke + Arcuate (Broca gelé)
    'wikipedia_long'  : Tous les modules, LR Broca × 0.1
    'oscar_filtered'  : Tous les modules, LR uniforme
    'oscar_full'      : Tous les modules + cosine annealing

Critères de passage (config.phase_thresholds) :
    - F_max : moyenne F sur 100 derniers steps < seuil
    - r_min : moyenne r sur 100 derniers steps > seuil
    - steps : minimum de steps avant d'évaluer les critères

Usage :
    from training.trainer import CurriculumTrainer
    trainer = CurriculumTrainer(wernicke, broca, arcuate, tokenizer, config)
    results = trainer.run(start_phase='bootstrap')
"""

import csv
import time
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from config import SNNConfig
from core.oscillator import OscillatoryClock
from graph.phase_sync import compute_order_parameter
from training.loss import phase_loss
from training.datasets import WikiFrDataset, OSCARFrDataset, build_dataloader


class CurriculumTrainer:
    """
    Trainer curriculum pour le framework SNN-PC Broca-Wernicke.

    Responsabilités :
    - Gestion du gel/dégel des modules selon la phase
    - Construction des datasets (Wikipedia FR ou OSCAR)
    - Boucle d'entraînement avec les 3 échelles temporelles imbriquées
    - Évaluation des critères de passage automatiques
    - Logging CSV et sauvegarde de checkpoints
    """

    PHASE_ORDER = [
        'bootstrap',
        'wikipedia_short',
        'wikipedia_long',
        'oscar_filtered',
        'oscar_full',
    ]

    def __init__(
        self,
        wernicke,
        broca,
        arcuate,
        tokenizer,
        config: SNNConfig,
    ):
        """
        Args:
            wernicke  : instance de WernickeModule
            broca     : instance de BrocaModule
            arcuate   : instance de ArcuateFasciculus
            tokenizer : CamembertTokenizer (ou compatible HuggingFace)
            config    : SNNConfig avec les champs de curriculum
        """
        self.wernicke  = wernicke
        self.broca     = broca
        self.arcuate   = arcuate
        self.tokenizer = tokenizer
        self.config    = config
        self.clock     = OscillatoryClock(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_to_device()

        # Initialisation du fichier de log CSV
        log_path = Path(self.config.log_csv_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_path, 'w', newline='', encoding='utf-8')
        self._csv = csv.DictWriter(self._log_file, fieldnames=[
            'phase', 'step', 'F_W', 'F_B', 'F_total',
            'r_W', 'r_B', 'sync_loss', 'lr', 'elapsed_s',
        ])
        self._csv.writeheader()

        # Historique glissant pour les critères de passage (reset entre phases)
        self._history: dict[str, list] = {'F': [], 'r': []}

        # Référence vers le scheduler (None si pas de cosine annealing)
        self.scheduler = None

    # ── Interface publique ─────────────────────────────────────────────────────

    def run(
        self,
        start_phase: str = 'bootstrap',
        end_phase: str = 'oscar_full',
    ) -> dict[str, dict]:
        """
        Lance le curriculum depuis start_phase jusqu'à end_phase inclus.

        Chaque phase :
        1. Configure le gel/dégel et l'optimizer
        2. Construit le dataset approprié
        3. Boucle d'entraînement jusqu'aux critères ou au max_steps
        4. Sauvegarde un checkpoint

        Args:
            start_phase : première phase à exécuter (incluse)
            end_phase   : dernière phase à exécuter (incluse)

        Returns:
            dict phase_name → {'steps', 'F_final', 'r_final'}
        """
        start_idx = self.PHASE_ORDER.index(start_phase)
        end_idx   = self.PHASE_ORDER.index(end_phase)
        results   = {}

        for phase in self.PHASE_ORDER[start_idx: end_idx + 1]:
            print(f"\n{'=' * 60}")
            print(f"  PHASE : {phase.upper()}")
            print(f"  Seuils : {self.config.phase_thresholds[phase]}")
            print(f"  Max tokens : {self.config.phase_max_tokens[phase]}")
            print(f"{'=' * 60}")

            self._configure_phase(phase)
            dataset = self._build_dataset(phase)
            loader  = build_dataloader(
                dataset,
                batch_size=self._batch_size_for(phase),
                shuffle=(phase != 'bootstrap'),   # bootstrap = ordre par longueur
                num_workers=0,                     # 0 pour éviter les forks problématiques
            )

            metrics = self._run_phase(phase, loader)
            results[phase] = metrics

            self._save_checkpoint(phase)
            self._history = {'F': [], 'r': []}    # reset pour la phase suivante

        self._log_file.close()
        return results

    # ── Configuration par phase ────────────────────────────────────────────────

    def _configure_phase(self, phase: str) -> None:
        """
        Gèle/dégèle les modules et instancie l'optimizer pour la phase courante.

        Principe de dégel progressif :
        - On commence toujours par tout geler (état propre)
        - On dégèle sélectivement selon la phase
        - Le LR de Broca est réduit lors de son premier dégel (wikipedia_long)
          pour éviter de détruire les représentations déjà apprises par Wernicke
        """
        # Geler tout d'abord — état propre
        self.wernicke.freeze()
        self.broca.freeze()
        self.arcuate.freeze()

        base_lr = self.config.lr_weights

        if phase == 'bootstrap':
            # Seul Wernicke est entraîné : apprentissage des représentations sémantiques
            self.wernicke.unfreeze()
            self.optimizer = optim.Adam(
                self.wernicke.parameters(), lr=base_lr
            )

        elif phase == 'wikipedia_short':
            # Wernicke + Arcuate : apprentissage du canal de communication
            self.wernicke.unfreeze()
            self.arcuate.unfreeze()
            self.optimizer = optim.Adam([
                {'params': self.wernicke.parameters(), 'lr': base_lr},
                {'params': self.arcuate.parameters(),  'lr': base_lr * 0.5},
            ])

        elif phase == 'wikipedia_long':
            # Tous les modules, mais Broca avec un LR réduit (dégel prudent)
            self.wernicke.unfreeze()
            self.arcuate.unfreeze()
            self.broca.unfreeze()
            self.optimizer = optim.Adam([
                {'params': self.wernicke.parameters(), 'lr': base_lr},
                {'params': self.arcuate.parameters(),  'lr': base_lr * 0.5},
                {'params': self.broca.parameters(),    'lr': base_lr * 0.1},
            ])

        elif phase in ('oscar_filtered', 'oscar_full'):
            # Entraînement complet, tous les modules au même LR
            self.wernicke.unfreeze()
            self.arcuate.unfreeze()
            self.broca.unfreeze()
            all_params = (
                list(self.wernicke.parameters()) +
                list(self.arcuate.parameters())  +
                list(self.broca.parameters())
            )
            self.optimizer = optim.Adam(all_params, lr=base_lr)

            # Cosine annealing pour les phases oscar (fine-tuning)
            max_steps = self.config.phase_thresholds[phase].get('steps')
            if max_steps:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max_steps, eta_min=base_lr * 0.01
                )
            else:
                self.scheduler = None

        # Afficher l'état de gel
        print(f"  Wernicke gelé : {self.wernicke.is_frozen()}")
        print(f"  Broca gelé    : {self.broca.is_frozen()}")
        print(f"  Arcuate gelé  : {self.arcuate.is_frozen()}")

    def _build_dataset(self, phase: str):
        """
        Instancie le dataset approprié à la phase.

        Phases wikipedia_* → WikiFrDataset
        Phases oscar_*     → OSCARFrDataset (avec ou sans filtres qualité)
        """
        max_tokens = self.config.phase_max_tokens[phase]

        if phase in ('bootstrap', 'wikipedia_short', 'wikipedia_long'):
            return WikiFrDataset(
                tokenizer=self.tokenizer,
                max_tokens=max_tokens,
                min_tokens=max(8, max_tokens // 4),
                length_curriculum=(phase == 'bootstrap'),
                cache_dir=self.config.data_cache_dir + '/wikipedia',
                config_name=self.config.wikipedia_config_name,
                max_articles=self.config.wiki_max_articles,
            )
        else:
            return OSCARFrDataset(
                tokenizer=self.tokenizer,
                source='occiglot',
                max_tokens=max_tokens,
                apply_quality_filter=(phase == 'oscar_filtered'),
                buffer_size=self.config.oscar_buffer_size,
                cache_dir=self.config.data_cache_dir + '/oscar',
            )

    def _batch_size_for(self, phase: str) -> int:
        """Taille de batch adaptée à la longueur de séquence de chaque phase."""
        return {
            'bootstrap':       16,
            'wikipedia_short': 32,
            'wikipedia_long':  16,
            'oscar_filtered':  16,
            'oscar_full':      8,
        }[phase]

    # ── Boucle d'entraînement ──────────────────────────────────────────────────

    def _run_phase(self, phase: str, loader) -> dict:
        """
        Boucle principale d'une phase du curriculum.

        Arrêt anticipé si les critères de passage (config.phase_thresholds)
        sont satisfaits après au moins 'steps' steps d'entraînement.

        Args:
            phase  : identifiant de la phase courante
            loader : DataLoader issu de _build_dataset

        Returns:
            dict {'steps', 'F_final', 'r_final'}
        """
        threshold = self.config.phase_thresholds[phase]
        max_steps = self.config.max_steps_override or threshold.get('steps')
        global_step = 0
        t0 = time.time()

        pbar = tqdm(total=max_steps, desc=phase, unit='step', dynamic_ncols=True)

        # Réinitialiser les états des modules pour la nouvelle phase
        self.wernicke.reset_state()
        self.broca.reset_state()
        self.arcuate.reset_state()
        self.clock.reset()

        for batch in loader:
            if max_steps and global_step >= max_steps:
                break

            # ── τ_learn : un pas d'optimisation ──────────────────────────────
            metrics = self._train_step(batch, phase)

            # Mise à jour de l'historique de convergence
            self._history['F'].append(metrics['F_total'])
            self._history['r'].append(metrics['r_mean'])

            # Logging CSV
            self._csv.writerow({
                'phase':     phase,
                'step':      global_step,
                'F_W':       round(metrics['F_W'], 6),
                'F_B':       round(metrics['F_B'], 6),
                'F_total':   round(metrics['F_total'], 6),
                'r_W':       round(metrics['r_W'], 6),
                'r_B':       round(metrics['r_B'], 6),
                'sync_loss': round(metrics.get('sync', 0.0), 6),
                'lr':        self.optimizer.param_groups[0]['lr'],
                'elapsed_s': round(time.time() - t0, 1),
            })
            self._log_file.flush()

            pbar.set_postfix({
                'F': f"{metrics['F_total']:.3f}",
                'r': f"{metrics['r_mean']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            pbar.update(1)
            global_step += 1

            # Critère de passage anticipé (évalué toutes les 100 steps)
            if global_step % 100 == 0 and self._can_advance(threshold, global_step):
                print(f"\n  ✓ Critères atteints après {global_step} steps — passage à la phase suivante")
                break

        pbar.close()

        # Métriques finales : moyenne des 100 derniers steps (ou tout si < 100)
        n_recent = min(100, len(self._history['F']))
        F_final = float(torch.tensor(self._history['F'][-n_recent:]).mean().item())
        r_final = float(torch.tensor(self._history['r'][-n_recent:]).mean().item())

        print(f"\n  Phase '{phase}' terminée : {global_step} steps | F={F_final:.4f} | r={r_final:.4f}")

        return {
            'steps':   global_step,
            'F_final': F_final,
            'r_final': r_final,
        }

    def _train_step(self, batch: dict, phase: str) -> dict:
        """
        Un step d'optimisation complet — implémente les 3 boucles imbriquées.

        Boucles :
            τ_learn  : ce niveau (appel de la fonction)
            τ_pred   : boucle sur n_inference_steps (inférence PC)
            τ_spike  : intégré dans LIFNeuron.forward() (pas dt)

        Args:
            batch : dict avec 'input_ids' (batch, seq_len) et 'attention_mask'
            phase : phase courante (pour la sélection de la loss)

        Returns:
            dict de métriques numériques (pour logging)
        """
        # Déplacer le batch sur le device
        input_ids = batch['input_ids'].to(self.device).long()
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        x_input = self._token_ids_to_bow(input_ids, attention_mask)

        self.optimizer.zero_grad()

        # ── τ_pred : boucle d'inférence PC ────────────────────────────────────
        gamma_phases = []
        epsilon_W_final = None
        epsilon_B_final = None

        for _ in range(self.config.n_inference_steps):
            # ── τ_spike : dynamique membranaire (intégrée dans LIFNeuron) ──────
            clock_state = self.clock.step()

            # Module Wernicke
            mu_prior = torch.zeros(
                x_input.shape[0], self.config.dim_wernicke,
                device=self.device
            )
            out_W = self.wernicke(
                x_input=x_input,
                mu_prior=mu_prior,
                clock=self.clock,
            )

            # Fascicule arqué W→B
            msg_W2B, _ = self.arcuate.transmit(
                message=out_W['prediction'],
                direction='W2B',
                visit_history=['W'],
            )

            # Module Broca (contexte = représentation nulle si non fourni)
            x_context = torch.zeros(
                x_input.shape[0], self.config.dim_broca,
                device=self.device
            )
            out_B = self.broca(
                mu_wernicke=msg_W2B,
                x_context=x_context,
                clock=self.clock,
            )

            gamma_phases.append(clock_state['phi_gamma'])
            epsilon_W_final = out_W['epsilon']
            epsilon_B_final = out_B['epsilon']

        # ── Calcul du paramètre d'ordre de Kuramoto ───────────────────────────
        phases_tensor = torch.tensor(gamma_phases, device=self.device)
        r_W = compute_order_parameter(phases_tensor)
        r_B = compute_order_parameter(phases_tensor)   # on utilise la même séquence de phases

        # ── Loss selon la phase ────────────────────────────────────────────────
        loss, breakdown = phase_loss(
            epsilon_W=epsilon_W_final,
            epsilon_B=epsilon_B_final,
            r_W=r_W,
            r_B=r_B,
            phase=phase,
        )

        # ── Rétropropagation ──────────────────────────────────────────────────
        loss.backward()

        # Clipping du gradient (essentiel pour la stabilité avec les SNNs)
        all_params = (
            list(self.wernicke.parameters()) +
            list(self.broca.parameters())    +
            list(self.arcuate.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            **breakdown,
            'F_total': float(loss.item()),
            'r_W':     float(r_W),
            'r_B':     float(r_B),
            'r_mean':  float((r_W + r_B) / 2.0),
        }

    def _can_advance(self, threshold: dict, current_steps: int) -> bool:
        """
        Vérifie les critères de passage sur les 100 derniers steps.

        Les critères ne sont évalués qu'après threshold['steps'] steps minimum,
        pour éviter un passage prématuré dû au bruit initial.

        Args:
            threshold     : dict avec 'F_max', 'r_min', 'steps'
            current_steps : nombre de steps effectués dans cette phase

        Returns:
            True si les deux critères sont satisfaits
        """
        min_steps = threshold.get('steps') or 0
        if current_steps < min_steps:
            return False
        if len(self._history['F']) < 100:
            return False

        recent_F = torch.tensor(self._history['F'][-100:]).mean().item()
        recent_r = torch.tensor(self._history['r'][-100:]).mean().item()

        F_ok = (threshold.get('F_max') is None) or (recent_F <= threshold['F_max'])
        r_ok = (threshold.get('r_min') is None) or (recent_r >= threshold['r_min'])

        return F_ok and r_ok

    # ── Checkpoints ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, phase: str) -> None:
        """Sauvegarde l'état de tous les modules à la fin d'une phase."""
        path = Path(self.config.checkpoint_dir) / f"{phase}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'phase':     phase,
            'wernicke':  self.wernicke.state_dict(),
            'broca':     self.broca.state_dict(),
            'arcuate':   self.arcuate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"  ✓ Checkpoint sauvegardé : {path}")

    def load_checkpoint(self, phase: str) -> None:
        """
        Charge l'état des modules depuis le checkpoint d'une phase.

        Args:
            phase : nom de la phase dont on charge le checkpoint
        """
        path = Path(self.config.checkpoint_dir) / f"{phase}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.wernicke.load_state_dict(ckpt['wernicke'])
        self.broca.load_state_dict(ckpt['broca'])
        self.arcuate.load_state_dict(ckpt['arcuate'])
        print(f"  ✓ Checkpoint chargé depuis : {path}")

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _move_to_device(self) -> None:
        """Déplace tous les modules sur le device configuré."""
        self.wernicke.to(self.device)
        self.broca.to(self.device)
        self.arcuate.to(self.device)

    def _token_ids_to_bow(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Convertit une séquence de token IDs en représentation sac-de-mots normalisée.

        Le module Wernicke attend un vecteur dense de taille vocabulaire.
        Pour l'entraînement sur corpus, on agrège donc les token IDs de chaque
        séquence en histogramme normalisé plutôt que de passer les IDs bruts.
        """
        batch_size = input_ids.shape[0]
        vocab_size = self.wernicke.input_projection.in_features

        if attention_mask is None:
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            attention_mask = input_ids.ne(pad_id).long()

        if input_ids.device.type == 'mps':
            cpu_input_ids = input_ids.detach().to('cpu')
            cpu_attention_mask = attention_mask.detach().to('cpu')
            cpu_bow = self._token_ids_to_bow(cpu_input_ids, cpu_attention_mask)
            return cpu_bow.to(self.device)

        valid_mask = attention_mask.bool()
        safe_ids = input_ids.masked_fill(~valid_mask, 0)

        bow = torch.zeros(batch_size, vocab_size, device=input_ids.device)
        bow.scatter_add_(1, safe_ids, valid_mask.float())

        lengths = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return bow / lengths
