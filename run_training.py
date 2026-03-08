"""
run_training.py — Point d'entrée CLI pour l'entraînement SNN-PC.

Lance le curriculum complet (ou partiel) sur Wikipedia FR + OSCAR/Occiglot.
Tous les hyperparamètres sont centralisés dans SNNConfig (config.py).

Usage :
    # Entraînement complet depuis le début
    python run_training.py

    # Démarrer à partir de la phase wikipedia_long
    python run_training.py --start wikipedia_long

    # Entraîner uniquement les phases oscar
    python run_training.py --start oscar_filtered --end oscar_full

    # Reprendre depuis un checkpoint (la phase start doit avoir un .pt)
    python run_training.py --start oscar_filtered --resume wikipedia_long

    # Générer seulement les graphiques depuis un CSV existant
    python run_training.py --plot-only

Arborescence générée :
    checkpoints/   — un .pt par phase terminée
    logs/          — training_log.csv (toutes les phases)
    results/       — dashboard.png, raster.png, curriculum_summary.png
"""

import argparse
import sys
from pathlib import Path


# ── Phases disponibles ────────────────────────────────────────────────────────
VALID_PHASES = [
    'bootstrap',
    'wikipedia_short',
    'wikipedia_long',
    'oscar_filtered',
    'oscar_full',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Entraînement curriculum SNN-PC Broca-Wernicke (Wikipedia FR + OSCAR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--start',
        choices=VALID_PHASES,
        default='bootstrap',
        help='Phase de départ du curriculum (défaut : bootstrap)',
    )
    parser.add_argument(
        '--end',
        choices=VALID_PHASES,
        default='oscar_full',
        help='Phase de fin du curriculum, incluse (défaut : oscar_full)',
    )
    parser.add_argument(
        '--resume',
        choices=VALID_PHASES,
        default=None,
        metavar='PHASE',
        help='Charger le checkpoint de la PHASE indiquée avant de démarrer',
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Générer uniquement les graphiques depuis le CSV existant (pas d\'entraînement)',
    )
    parser.add_argument(
        '--log-csv',
        default=None,
        metavar='PATH',
        help='Chemin du fichier CSV de log (remplace config.log_csv_path)',
    )
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        metavar='DIR',
        help='Répertoire des checkpoints (remplace config.checkpoint_dir)',
    )
    parser.add_argument(
        '--results-dir',
        default='./results',
        metavar='DIR',
        help='Répertoire de sauvegarde des figures (défaut : ./results)',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        metavar='N',
        help='Borne dure sur le nombre de steps par phase pour un entraînement court ou de validation',
    )
    parser.add_argument(
        '--wiki-max-articles',
        type=int,
        default=None,
        metavar='N',
        help='Limite le nombre d\'articles Wikipedia consommés pour éviter un chargement massif',
    )
    parser.add_argument(
        '--oscar-buffer-size',
        type=int,
        default=None,
        metavar='N',
        help='Taille du buffer local OSCAR utilisé au démarrage',
    )
    args = parser.parse_args()

    # Validation : start doit précéder end dans le curriculum
    if VALID_PHASES.index(args.start) > VALID_PHASES.index(args.end):
        parser.error(
            f"--start '{args.start}' doit précéder --end '{args.end}' "
            f"dans l'ordre du curriculum."
        )

    return args


def build_model(config, vocab_size: int):
    """
    Instancie les trois modules du réseau Broca-Wernicke.

    Args:
        config : SNNConfig avec dim_wernicke, dim_broca, dim_arcuate
        vocab_size : taille réelle du vocabulaire du tokenizer chargé

    Returns:
        (wernicke, broca, arcuate) — modules nn.Module prêts à l'entraînement
    """
    from modules.wernicke import WernickeModule
    from modules.broca   import BrocaModule
    from modules.arcuate import ArcuateFasciculus

    wernicke = WernickeModule(vocab_size=vocab_size, config=config)
    broca    = BrocaModule(vocab_size=vocab_size,   config=config)
    arcuate  = ArcuateFasciculus(config=config)

    n_params = sum(p.numel() for p in wernicke.parameters())
    n_params += sum(p.numel() for p in broca.parameters())
    n_params += sum(p.numel() for p in arcuate.parameters())
    print(f"  Paramètres totaux : {n_params:,}")

    return wernicke, broca, arcuate


def load_tokenizer(config):
    """
    Charge le tokenizer CamemBERT depuis HuggingFace Hub.

    Le tokenizer est mis en cache dans config.data_cache_dir pour éviter
    les téléchargements répétés entre les phases.

    Args:
        config : SNNConfig avec tokenizer_name et data_cache_dir

    Returns:
        CamembertTokenizer prêt à l'emploi
    """
    try:
        from transformers import CamembertTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers est requis : pip install transformers>=4.38 sentencepiece>=0.1.99"
        ) from e

    print(f"  Chargement du tokenizer '{config.tokenizer_name}'...")
    tokenizer = CamembertTokenizer.from_pretrained(
        config.tokenizer_name,
        cache_dir=config.data_cache_dir + '/tokenizer',
    )
    print(f"  Taille du vocabulaire : {tokenizer.vocab_size}")
    return tokenizer


def generate_plots(log_csv_path: str, results_dir: str) -> None:
    """
    Génère et sauvegarde les figures de suivi depuis le fichier CSV de log.

    Args:
        log_csv_path : chemin vers le CSV produit par CurriculumTrainer
        results_dir  : répertoire de destination pour les PNG
    """
    from training.viz import plot_training_dashboard

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if not Path(log_csv_path).exists():
        print(f"  ⚠ Fichier de log introuvable : {log_csv_path}")
        print("    Pas de graphique généré.")
        return

    print(f"\n  Génération du dashboard depuis : {log_csv_path}")
    dashboard_path = str(results_path / 'dashboard.png')
    fig = plot_training_dashboard(log_csv_path, save_path=dashboard_path)

    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def generate_summary_plot(results: dict, results_dir: str) -> None:
    """
    Génère le graphique récapitulatif des métriques finales par phase.

    Args:
        results     : dict retourné par trainer.run()
        results_dir : répertoire de destination
    """
    from training.viz import plot_phase_curriculum_summary
    import matplotlib.pyplot as plt

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    summary_path = str(results_path / 'curriculum_summary.png')
    fig = plot_phase_curriculum_summary(results, save_path=summary_path)
    plt.close(fig)


def main() -> int:
    """
    Point d'entrée principal.

    Returns:
        Code de retour : 0 = succès, 1 = erreur
    """
    args = parse_args()

    print("\n" + "=" * 60)
    print("  SNN-PC Hybrid — Curriculum Broca-Wernicke")
    print("  Wikipedia FR + OSCAR/Occiglot (français)")
    print("=" * 60)

    # ── Chargement de la configuration ───────────────────────────────────────
    from config import SNNConfig
    config = SNNConfig()

    # Surcharges CLI (chemin de log et de checkpoints)
    if args.log_csv:
        config.log_csv_path = args.log_csv
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        config.max_steps_override = args.max_steps
    if args.wiki_max_articles is not None:
        config.wiki_max_articles = args.wiki_max_articles
    if args.oscar_buffer_size is not None:
        config.oscar_buffer_size = args.oscar_buffer_size

    print(f"\n  Device      : {config.device}")
    print(f"  Log CSV     : {config.log_csv_path}")
    print(f"  Checkpoints : {config.checkpoint_dir}")
    print(f"  Phases      : {args.start} → {args.end}")

    # ── Mode visualisation uniquement ─────────────────────────────────────────
    if args.plot_only:
        print("\n  Mode --plot-only : génération des graphiques depuis le CSV existant.")
        generate_plots(config.log_csv_path, args.results_dir)
        return 0

    # ── Chargement du tokenizer ───────────────────────────────────────────────
    print("\n[1/4] Tokenizer")
    tokenizer = load_tokenizer(config)

    # ── Construction du modèle ────────────────────────────────────────────────
    print("\n[2/4] Modèle")
    wernicke, broca, arcuate = build_model(config, tokenizer.vocab_size)

    # ── Instanciation du trainer ──────────────────────────────────────────────
    print("\n[3/4] Trainer")
    from training.trainer import CurriculumTrainer
    trainer = CurriculumTrainer(
        wernicke=wernicke,
        broca=broca,
        arcuate=arcuate,
        tokenizer=tokenizer,
        config=config,
    )

    # Reprise depuis un checkpoint existant
    if args.resume is not None:
        print(f"\n  Chargement du checkpoint '{args.resume}'...")
        try:
            trainer.load_checkpoint(args.resume)
        except FileNotFoundError as e:
            print(f"  ✗ Erreur : {e}")
            return 1

    # ── Curriculum d'entraînement ─────────────────────────────────────────────
    print(f"\n[4/4] Curriculum : {args.start} → {args.end}")
    try:
        results = trainer.run(start_phase=args.start, end_phase=args.end)
    except KeyboardInterrupt:
        print("\n\n  Entraînement interrompu par l'utilisateur (Ctrl+C).")
        print("  Les checkpoints déjà sauvegardés restent disponibles.")
        return 1

    # ── Rapport final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Résultats finaux par phase :")
    print("=" * 60)
    for phase, metrics in results.items():
        print(
            f"  {phase:<20}  "
            f"steps={metrics['steps']:>7,}  "
            f"F={metrics['F_final']:.4f}  "
            f"r={metrics['r_final']:.4f}"
        )

    # ── Génération des figures ────────────────────────────────────────────────
    print(f"\n  Génération des figures → {args.results_dir}/")
    generate_plots(config.log_csv_path, args.results_dir)
    generate_summary_plot(results, args.results_dir)

    print("\n  ✓ Entraînement terminé avec succès.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
