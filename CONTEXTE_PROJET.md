# Contexte du projet SNN-PC Hybrid

## Résumé

Ce dépôt implémente un framework expérimental qui combine :

- des neurones impulsionnels de type LIF,
- du Predictive Coding,
- une synchronisation de phase de type Kuramoto,
- une architecture inspirée des aires de Wernicke et Broca.

Le projet sert à deux usages distincts :

- un mode expérimental léger, reproductible localement, via des expériences jouets et des tests unitaires ;
- un mode entraînement plus ambitieux, orienté corpus français réels, via HuggingFace et un curriculum en plusieurs phases.

## Objectif technique

L'idée centrale est de représenter la sémantique et la syntaxe comme des états latents continus mis à jour par minimisation d'erreurs de prédiction, puis de faire circuler ces états via des mécanismes inspirés du spike coding et de la synchronisation oscillatoire.

Le dépôt ne se présente pas comme une application produit finie. C'est une base de recherche/prototypage avec :

- des composants neuronaux réutilisables,
- des modules cognitifs spécialisés,
- des expériences ciblées,
- une boucle d'entraînement plus lourde pour des corpus externes.

## Structure logique

### Noyau biologique et signal

- `core/neuron.py` : neurone LIF avec terme d'erreur de Predictive Coding injecté dans la dynamique membranaire.
- `core/encoding.py` : schémas d'encodage en spikes, dont `rate_encode`, `phase_encode` et `burst_encode`.
- `core/oscillator.py` : horloge thêta/gamma qui fournit les phases et l'amplitude gamma modulée.
- `core/synapse.py` : logique synaptique/STDP compatible avec le reste du framework.

### Modules cognitifs

- `modules/wernicke.py` : encodeur sémantique. Reçoit un input, minimise une erreur de prédiction, puis projette un message vers le canal arcué.
- `modules/broca.py` : décodeur syntaxique. Reçoit un message sémantique, l'aligne avec un contexte local, puis génère des logits sur le vocabulaire.
- `modules/arcuate.py` : canal de communication Wernicke ↔ Broca, avec amortissement cyclique exponentiel pour limiter les boucles récurrentes.

### Graphe et convergence

- `graph/pc_gnn.py` : extension du Predictive Coding à un graphe arbitraire, avec messages top-down et bottom-up.
- `graph/phase_sync.py` : outils de convergence de type Kuramoto, calcul du paramètre d'ordre et mise à jour des phases.
- `graph/message_passing.py` : variante orientée passage de messages, couverte par les tests.

### Entraînement et données

- `training/trainer.py` : orchestrateur principal du curriculum d'entraînement sur cinq phases.
- `training/datasets.py` : datasets HuggingFace pour Wikipedia FR et OSCAR/Occiglot, avec chunking, filtrage et streaming.
- `training/loss.py` : énergie libre variationnelle et pertes associées.
- `training/surrogate.py` : surrogate gradient pour la fonction de spike.
- `training/viz.py` : génération des graphiques de suivi.

### Expériences et validation

- `experiments/toy_language.py` : expérience jouet sur séquences sujet-verbe-objet.
- `experiments/convergence_test.py` : comparaison de la convergence sur graphes avec et sans amortissement cyclique.
- `tests/` : tests unitaires sur le neurone, l'encodage, la convergence et les composants graphe.
- `demo.ipynb` : notebook interactif pour exploration/démonstration.

## Flux d'exécution importants

### 1. Expérience jouet

Le scénario le plus simple pour comprendre le projet est `experiments/toy_language.py`.

Flux global :

1. un sujet est encodé en one-hot ;
2. Wernicke construit une représentation sémantique ;
3. le fascicule arqué transmet un message vers Broca ;
4. Broca génère des logits de vocabulaire ;
5. le tout est évalué via énergie libre, cohérence de phase et précision token.

Ce script est le meilleur point d'entrée pour comprendre le comportement du modèle sans dépendre d'un pipeline de données massif.

### 2. Entraînement corpus français

`run_training.py` est le vrai point d'entrée CLI utile du dépôt.

Il :

- charge `SNNConfig`,
- instancie Wernicke, Broca et Arcuate,
- charge le tokenizer CamemBERT,
- construit le `CurriculumTrainer`,
- entraîne sur un curriculum `bootstrap -> wikipedia_short -> wikipedia_long -> oscar_filtered -> oscar_full`,
- enregistre logs, checkpoints et figures.

Ce flux dépend fortement de HuggingFace, de jeux de données volumineux et d'un environnement réellement préparé.

### 3. Point d'entrée principal du dépôt

`main.py` n'est pas un point d'entrée fonctionnel pour le framework. Il ne fait qu'afficher `Hello from snn-pc-hybrid!`.

En pratique, les vrais points d'entrée sont :

- `run_training.py`,
- `experiments/toy_language.py`,
- `experiments/convergence_test.py`,
- `demo.ipynb`.

## Configuration centrale

`config.py` contient la dataclass `SNNConfig`, qui centralise :

- dynamique membranaire : `tau_m`, `tau_syn`, `v_rest`, `v_threshold`, `v_reset`, `dt` ;
- oscillations : `theta_freq`, `gamma_freq` ;
- Predictive Coding : `eta_pc`, `sigma_prior`, `n_inference_steps` ;
- STDP : `A_plus`, `A_minus`, `tau_plus`, `tau_minus`, `lr_weights` ;
- tailles de modules : `dim_wernicke`, `dim_broca`, `dim_arcuate` ;
- stabilité : `gamma_stability`, `cycle_damping_lambda` ;
- curriculum : phases, seuils de passage, longueurs maximales de séquence ;
- chemins d'E/S : cache, checkpoints et logs.

Cette classe est le centre de gravité du dépôt. Toute analyse ou modification du comportement global devrait commencer ici.

## Dépendances

Deux niveaux de dépendances coexistent.

### Dépendances de base

Dans `pyproject.toml` :

- `torch`
- `torch-geometric`
- `numpy`
- `matplotlib`
- `tqdm`
- `scipy`
- `networkx`
- `jupyter`
- `ipykernel`

### Dépendances d'entraînement étendu

Dans `requirements.txt`, le dépôt ajoute aussi :

- `datasets`
- `transformers`
- `sentencepiece`
- `pandas`
- `seaborn`
- `hf_transfer`

Conclusion pratique : `pyproject.toml` décrit une partie utile du socle, mais `requirements.txt` reflète mieux le besoin réel pour le pipeline d'entraînement complet.

## État actuel observé

### Ce qui est solide

- L'architecture est cohérente entre README, configuration et code.
- Les composants principaux sont bien séparés par responsabilité.
- Le dépôt est déjà testable localement.
- La suite de tests passe intégralement dans l'environnement du projet : `46 passed`.

### Ce qui ressemble encore à un dépôt de recherche

- `main.py` est encore un stub.
- Le pipeline d'entraînement réel dépend de téléchargements lourds et de sources externes.
- Une partie importante du comportement est documentée de manière théorique, parfois plus ambitieuse que ce qui est nécessaire pour une exécution locale simple.
- Le canal Arcuate reconstruit des messages depuis des spikes stochastiques ; c'est un choix expérimental plutôt qu'une implémentation "production".

### Écarts et points d'attention

- Le README annonce Python `>=3.10`, alors que `pyproject.toml` impose `>=3.12`.
- `requirements.txt` est plus complet que `pyproject.toml` pour l'entraînement sur corpus.
- Le dépôt mélange une logique de démonstration légère et une logique d'entraînement lourde ; il faut choisir explicitement le mode d'usage selon l'objectif.

## Recommandation d'usage

Pour explorer le dépôt rapidement :

1. lire `config.py` ;
2. lire `experiments/toy_language.py` ;
3. examiner `modules/wernicke.py`, `modules/broca.py` et `modules/arcuate.py` ;
4. vérifier `training/trainer.py` seulement si l'objectif est l'entraînement réel ;
5. utiliser les tests pour valider toute modification locale.

## Lecture synthétique du projet

En une phrase : ce dépôt est un framework de recherche en neuro-inspiration computationnelle, structuré proprement, avec une base locale fiable pour les expériences et les tests, et un pipeline d'entraînement plus ambitieux mais plus coûteux et dépendant de ressources externes.