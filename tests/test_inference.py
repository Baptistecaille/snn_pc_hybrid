"""Tests de non-régression pour l'inférence batchée."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import SNNConfig
from run_training import build_model
from training.trainer import CurriculumTrainer


class DummyTokenizer:
    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]

        token_rows = []
        for text in texts:
            token_ids = [((ord(char) % (self.vocab_size - 1)) + 1) for char in text]
            if max_length is not None:
                token_ids = token_ids[:max_length]
            if not token_ids:
                token_ids = [1]
            token_rows.append(token_ids)

        width = max(len(row) for row in token_rows)
        padded = []
        masks = []
        for row in token_rows:
            pad_len = width - len(row)
            padded.append(row + [self.pad_token_id] * pad_len)
            masks.append([1] * len(row) + [0] * pad_len)

        return {
            'input_ids': torch.tensor(padded, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        return [f'tok_{token_id}' for token_id in ids]


def build_test_trainer(tmp_path):
    config = SNNConfig(
        dim_wernicke=16,
        dim_broca=16,
        dim_arcuate=8,
        n_inference_steps=3,
        device=torch.device('cpu'),
        log_csv_path=str(tmp_path / 'logs' / 'training.csv'),
        checkpoint_dir=str(tmp_path / 'checkpoints'),
    )
    tokenizer = DummyTokenizer(vocab_size=32)
    torch.manual_seed(0)
    wernicke, broca, arcuate = build_model(config, tokenizer.vocab_size)
    return CurriculumTrainer(wernicke, broca, arcuate, tokenizer, config)


def test_infer_batch_returns_one_result_per_text(tmp_path):
    trainer = build_test_trainer(tmp_path)

    results = trainer.infer_batch(['bonjour', 'salut'], top_k=4, use_mixed_precision=False)

    assert results['texts'] == ['bonjour', 'salut']
    assert len(results['top_ids']) == 2
    assert len(results['top_tokens']) == 2
    assert all(len(ids) == 4 for ids in results['top_ids'])
    assert all(len(tokens) == 4 for tokens in results['top_tokens'])
    assert trainer.wernicke.mu_W.shape[0] == 2
    assert trainer.broca.mu_B.shape[0] == 2


def test_batched_inference_matches_sequential_predictions(tmp_path):
    trainer = build_test_trainer(tmp_path)

    def deterministic_transmit(message, direction, visit_history):
        if direction == 'W2B':
            projected = trainer.arcuate.W2B_projection(message)
        else:
            projected = trainer.arcuate.B2W_projection(message)
        return torch.tanh(projected), 1.0

    trainer.arcuate.transmit = deterministic_transmit

    batched = trainer.infer_batch(['alpha', 'beta'], top_k=3, use_mixed_precision=False)
    first = trainer.infer_batch(['alpha'], top_k=3, use_mixed_precision=False)
    second = trainer.infer_batch(['beta'], top_k=3, use_mixed_precision=False)

    assert batched['top_ids'][0] == first['top_ids'][0]
    assert batched['top_ids'][1] == second['top_ids'][0]