"""
training/datasets.py — Loaders pour Wikipedia FR et OSCAR/Occiglot.

Deux datasets, une interface commune compatible torch.utils.data :
- WikiFrDataset  : corpus encyclopédique propre (phases bootstrap/wikipedia_*)
- OSCARFrDataset : corpus web filtré (phases oscar_*)

Dépendances (à installer) :
    pip install datasets>=2.18 transformers>=4.38 sentencepiece>=0.1.99
    # Optionnel — accélère les téléchargements HuggingFace :
    pip install hf_transfer>=0.1.6
    HF_HUB_ENABLE_HF_TRANSFER=1 python run_training.py

Architecture de chunking (WikiFrDataset) :
    Article texte brut → tokenisation → fenêtres non-chevauchantes de max_tokens tokens
    Les chunks trop courts (< min_tokens) sont silencieusement ignorés.
    Si length_curriculum=True, les chunks sont triés par longueur croissante
    pour présenter les exemples courts en premier (curriculum intra-phase).
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


WIKIPEDIA_DATASET_ID = os.environ.get('SNN_WIKIPEDIA_DATASET_ID', 'wikimedia/wikipedia')


def _build_hf_download_config():
    """Construit une configuration de telechargement HF si des timeouts sont fournis."""
    timeout = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')
    etag_timeout = os.environ.get('HF_HUB_ETAG_TIMEOUT')

    if timeout is None and etag_timeout is None:
        return None

    try:
        from datasets import DownloadConfig
    except ImportError:
        return None

    kwargs = {}
    if timeout is not None:
        try:
            kwargs['download_timeout'] = float(timeout)
        except ValueError:
            pass
    if etag_timeout is not None:
        try:
            kwargs['etag_timeout'] = float(etag_timeout)
        except ValueError:
            pass

    return DownloadConfig(**kwargs) if kwargs else None


def _cached_wikipedia_configs(cache_dir: str | None) -> list[str]:
    """Liste les configs Wikipedia FR deja presentes dans le cache local."""
    if not cache_dir:
        return []

    root = Path(cache_dir)
    candidates = [root / 'wikipedia', root]
    configs: set[str] = set()

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue

        for child in candidate.iterdir():
            if child.is_dir() and child.name.endswith('.fr'):
                configs.add(child.name)

    return sorted(configs)


def _resolve_wikipedia_config(preferred_config: str, cache_dir: str | None = None) -> str:
    """Retourne une config Wikipedia FR exploitable, en preferant le cache local."""
    cached_configs = _cached_wikipedia_configs(cache_dir)
    if preferred_config in cached_configs:
        return preferred_config

    try:
        from datasets import get_dataset_config_names
    except ImportError:
        config_names = []
    else:
        try:
            config_names = get_dataset_config_names(WIKIPEDIA_DATASET_ID)
        except Exception:
            config_names = []

    if preferred_config in config_names:
        return preferred_config

    if cached_configs:
        return cached_configs[-1]

    fr_configs = sorted(name for name in config_names if name.endswith('.fr'))
    if fr_configs:
        return fr_configs[-1]

    return preferred_config


class WikiFrDataset(Dataset):
    """
    Wikipedia français — corpus encyclopédique propre.

    Utilisé en phases 'bootstrap', 'wikipedia_short' et 'wikipedia_long'.
    Le chunking non-chevauchant garantit que chaque token est vu au plus une
    fois par époque, sans fuite d'information entre chunks.

    Chargement :
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        ds = WikiFrDataset(tokenizer, max_tokens=64)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

    Note : le premier appel déclenche le téléchargement depuis HuggingFace Hub
    (~20 Go pour Wikipedia FR). Les données sont cachées dans cache_dir.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens: int = 64,
        min_tokens: int = 16,
        length_curriculum: bool = True,
        cache_dir: str = './cache/wikipedia',
        split: str = 'train',
        config_name: str = '20231101.fr',
        max_articles: int | None = None,
    ):
        """
        Args:
            tokenizer         : CamembertTokenizer (ou tout tokenizer HuggingFace)
            max_tokens        : longueur maximale d'un chunk (en tokens)
            min_tokens        : longueur minimale pour qu'un chunk soit conservé
            length_curriculum : si True, trier par longueur croissante
                                (exemples courts en premier = curriculum intra-phase)
            cache_dir         : répertoire de cache HuggingFace
            split             : 'train' (seul split disponible pour Wikipedia FR)
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.max_articles = max_articles

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets est requis : pip install datasets>=2.18"
            ) from e

        resolved_config = _resolve_wikipedia_config(config_name, cache_dir)
        streaming = max_articles is not None

        if resolved_config != config_name:
            print(
                f"Config Wikipedia demandee '{config_name}' indisponible; utilisation de '{resolved_config}'"
            )

        print(
            f"Chargement Wikipedia FR (config={resolved_config}, split={split}, cache={cache_dir}, streaming={streaming})..."
        )
        load_kwargs = {
            'split': split,
            'cache_dir': cache_dir,
            'streaming': streaming,
        }
        download_config = _build_hf_download_config()
        if download_config is not None:
            load_kwargs['download_config'] = download_config

        raw = load_dataset(
            WIKIPEDIA_DATASET_ID,
            resolved_config,
            **load_kwargs,
        )

        # Tokenisation et chunking de tous les articles
        print(f"Chunking des articles (max_tokens={max_tokens}, min_tokens={min_tokens})...")
        self.chunks = self._build_chunks(raw, max_tokens, min_tokens, max_articles=max_articles)
        print(f"  → {len(self.chunks)} chunks produits")

        # Curriculum intra-phase : trier par longueur croissante
        if length_curriculum:
            self.chunks.sort(key=lambda x: x['length'])

    def _build_chunks(
        self,
        raw_dataset,
        max_tokens: int,
        min_tokens: int,
        max_articles: int | None = None,
    ) -> list:
        """
        Découpe chaque article en chunks de max_tokens tokens (non-chevauchants).

        Stratégie de padding : les chunks partiels (dernier chunk d'un article)
        sont paddés avec pad_token_id jusqu'à max_tokens, et un masque d'attention
        binaire est stocké.

        Args:
            raw_dataset : dataset HuggingFace itérable
            max_tokens  : taille de fenêtre
            min_tokens  : taille minimale pour retenir un chunk

        Returns:
            liste de dicts {'input_ids': Tensor, 'attention_mask': Tensor, 'length': int, 'title': str}
        """
        chunks = []
        pad_id = self.tokenizer.pad_token_id
        article_count = 0

        for article in raw_dataset:
            if max_articles is not None and article_count >= max_articles:
                break

            text = article.get('text', '')
            if not text:
                continue

            article_count += 1

            tokens = self.tokenizer(
                text,
                truncation=False,
                return_tensors='pt',
                add_special_tokens=False,
                verbose=False,
            )['input_ids'][0]   # (n_tokens,)

            # Découpage en fenêtres non-chevauchantes
            for start in range(0, max(1, len(tokens) - min_tokens + 1), max_tokens):
                chunk_ids = tokens[start: start + max_tokens]
                chunk_len = len(chunk_ids)

                if chunk_len < min_tokens:
                    continue

                # Padding si chunk partiel
                if chunk_len < max_tokens:
                    pad = torch.full((max_tokens - chunk_len,), pad_id, dtype=torch.long)
                    chunk_ids = torch.cat([chunk_ids, pad])

                mask = (chunk_ids != pad_id).long()

                chunks.append({
                    'input_ids':      chunk_ids,
                    'attention_mask': mask,
                    'length':         int(mask.sum().item()),
                    'title':          article.get('title', ''),
                })

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        item = self.chunks[idx]
        return {
            'input_ids':      item['input_ids'],
            'attention_mask': item['attention_mask'],
            'length':         item['length'],
        }


class OSCARFrDataset(Dataset):
    """
    OSCAR / Occiglot FineWeb — corpus web français filtré.

    Utilisé en phases 'oscar_filtered' (avec filtres qualité) et 'oscar_full'
    (sans filtres, volume maximal).

    Sources disponibles (par ordre de qualité décroissante) :
        'occiglot' → occiglot/occiglot-fineweb-v1.0   (recommandé)
        'oscar'    → oscar-corpus/OSCAR-2301
        'common'   → PleIAs/common_corpus

    Filtres qualité appliqués si apply_quality_filter=True :
        - Longueur ≥ 100 caractères
        - Ratio chiffres ≤ 30 %
        - Présence d'au moins un signe de ponctuation française

    Le dataset utilise le streaming HuggingFace pour éviter de charger
    plusieurs centaines de Go en RAM. Un buffer local de buffer_size exemples
    est rempli au démarrage et peut être renouvelé entre les époques via
    refresh_buffer().
    """

    def __init__(
        self,
        tokenizer,
        source: str = 'occiglot',
        max_tokens: int = 256,
        apply_quality_filter: bool = True,
        buffer_size: int = 10_000,
        cache_dir: str = './cache/oscar',
    ):
        """
        Args:
            tokenizer            : CamembertTokenizer
            source               : 'occiglot' | 'oscar' | 'common'
            max_tokens           : longueur maximale de séquence (en tokens)
            apply_quality_filter : activer les filtres qualité (phase oscar_filtered)
            buffer_size          : nombre d'exemples dans le buffer local
            cache_dir            : répertoire de cache HuggingFace
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.apply_quality_filter = apply_quality_filter
        self._buffer: list[dict] = []
        self._buffer_target = buffer_size

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets est requis : pip install datasets>=2.18"
            ) from e

        # Mapping source → (nom HuggingFace, kwargs additionnels)
        source_map = {
            'occiglot': ('occiglot/occiglot-fineweb-v1.0', {'language': 'fr'}),
            'oscar':    ('oscar-corpus/OSCAR-2301',         {'language': 'fr'}),
            'common':   ('PleIAs/common_corpus',            {'language': 'fr'}),
        }
        if source not in source_map:
            raise ValueError(f"source doit être l'un de {list(source_map.keys())}")

        hf_name, hf_kwargs = source_map[source]

        print(f"Initialisation du stream OSCAR ({source}) depuis HuggingFace...")
        self._stream = load_dataset(
            hf_name,
            split='train',
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **hf_kwargs,
        ).shuffle(buffer_size=min(buffer_size, 50_000))

        self._fill_buffer()

    def _quality_filter(self, text: str) -> bool:
        """
        Filtre qualité pour exclure les documents bruités.

        Critères (inspirés des heuristiques de C4/FineWeb) :
        - Longueur ≥ 100 caractères
        - Ratio chiffres ≤ 30 % (exclut les tableaux, listes de nombres)
        - Présence d'au moins un signe de ponctuation française
          (., !, ?, ;, :, «, », ', ", …)

        Args:
            text : texte brut du document

        Returns:
            True si le document passe tous les filtres
        """
        if len(text) < 100:
            return False

        digit_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
        if digit_ratio > 0.30:
            return False

        fr_punctuation = set('.,!?;:«»\u2018\u2019\u201c\u201d\u2026')
        if not any(c in fr_punctuation for c in text):
            return False

        return True

    def _fill_buffer(self) -> None:
        """
        Remplit le buffer local depuis le stream HuggingFace.

        Itère le stream jusqu'à atteindre buffer_target exemples valides.
        Si apply_quality_filter=True, les documents rejetés sont silencieusement ignorés.
        """
        count = 0
        print(f"Remplissage du buffer OSCAR ({self._buffer_target} exemples)...")

        for doc in self._stream:
            text = doc.get('text', doc.get('content', ''))
            if not text:
                continue
            if self.apply_quality_filter and not self._quality_filter(text):
                continue

            # Tokenisation avec troncature à max_tokens
            tokens = self.tokenizer(
                text[:4096],                # limite préventive contre les OOM
                truncation=True,
                max_length=self.max_tokens,
                padding='max_length',
                return_tensors='pt',
                verbose=False,
            )
            self._buffer.append({
                'input_ids':      tokens['input_ids'][0],
                'attention_mask': tokens['attention_mask'][0],
            })

            count += 1
            if count >= self._buffer_target:
                break

        print(f"  → {len(self._buffer)} exemples dans le buffer")

    def refresh_buffer(self) -> None:
        """
        Renouvelle le buffer local depuis le stream.

        À appeler entre les époques pour exposer de nouvelles données au modèle
        (le stream HuggingFace produit des données quasi-infinies via shuffle).
        """
        self._buffer.clear()
        self._fill_buffer()

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> dict:
        return self._buffer[idx]


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Wrapper commun pour créer un DataLoader depuis WikiFrDataset ou OSCARFrDataset.

    Args:
        dataset     : instance de WikiFrDataset ou OSCARFrDataset
        batch_size  : taille des mini-batches
        shuffle     : mélanger les données (False pour la phase bootstrap
                      où l'ordre par longueur est important)
        num_workers : nombre de workers de chargement parallèle

    Returns:
        DataLoader prêt à l'emploi
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,     # évite les batches partiels qui perturbent la norm du gradient
    )
