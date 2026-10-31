"""FT-Transformer (Gorishniy et al., 2021) для регрессии log1p(Valuation).

Feature Tokenizer:
    число j      → x_j · W_j + b_j;  при пропуске — обучаемый эмбеддинг «пропуск» M_j
    категория j  → Embedding_j[code] + b_j
    текст        → Linear(64 → d)  (один токен для SVD-представления Work Description)
    [CLS]        → обучаемый вектор, добавляется в начало последовательности
Далее N блоков Transformer (pre-LN: MHSA и FFN с ReGLU, dropout), голова по [CLS]:
LayerNorm → ReLU → Linear(d → 1). Модель предсказывает стандартизованный log1p(y).

Опции для экспериментов (по умолчанию выключены, baseline не меняется):
    txt_groups = G > 1   SVD-вектор режется на G равных частей, у каждой своя проекция → G токенов
    word_vocab, word_len слова WORK_DESC как токены: Embedding(слово) + позиционный эмбеддинг;
                         PAD (код 0) маскируется в attention
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HP = {
    "d_token": 192,
    "n_blocks": 3,
    "n_heads": 8,
    "d_ffn": 256,
    "attn_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
}


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: list[int], txt_dim: int, d: int,
                 txt_groups: int = 1, word_vocab: int = 0, word_len: int = 0):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(cat_cardinalities)
        self.txt_groups, self.word_len = txt_groups, (word_len if word_vocab else 0)
        s = 1 / math.sqrt(d)
        self.num_w = nn.Parameter(torch.empty(n_num, d).uniform_(-s, s))
        self.num_b = nn.Parameter(torch.empty(n_num, d).uniform_(-s, s))
        self.num_missing = nn.Parameter(torch.empty(n_num, d).uniform_(-s, s))
        offsets = torch.tensor([0] + list(cat_cardinalities[:-1])).cumsum(0)
        self.register_buffer("cat_offsets", offsets, persistent=False)
        self.cat_emb = nn.Embedding(sum(cat_cardinalities), d)
        nn.init.uniform_(self.cat_emb.weight, -s, s)
        self.cat_b = nn.Parameter(torch.empty(self.n_cat, d).uniform_(-s, s))
        if txt_groups == 1:
            self.txt = nn.Linear(txt_dim, d)
        else:
            assert txt_dim % txt_groups == 0
            k = txt_dim // txt_groups
            sk = 1 / math.sqrt(k)  # как у nn.Linear по умолчанию
            self.txt_w = nn.Parameter(torch.empty(txt_groups, k, d).uniform_(-sk, sk))
            self.txt_b = nn.Parameter(torch.empty(txt_groups, d).uniform_(-sk, sk))
        self.cls = nn.Parameter(torch.empty(1, 1, d).uniform_(-s, s))
        if word_vocab:
            self.word_emb = nn.Embedding(word_vocab, d, padding_idx=0)
            nn.init.uniform_(self.word_emb.weight, -s, s)
            with torch.no_grad():
                self.word_emb.weight[0].zero_()
            self.word_pos = nn.Parameter(torch.empty(word_len, d).uniform_(-s, s))

    @property
    def n_tokens(self) -> int:
        return 1 + self.n_num + self.n_cat + self.txt_groups + self.word_len

    def forward(self, num, num_mask, cat, txt, words=None):
        t_num = num.unsqueeze(-1) * self.num_w + self.num_b
        t_num = torch.where(num_mask.unsqueeze(-1), self.num_missing.expand_as(t_num), t_num)
        t_cat = self.cat_emb(cat + self.cat_offsets) + self.cat_b
        if self.txt_groups == 1:
            t_txt = self.txt(txt).unsqueeze(1)
        else:
            g = txt.view(txt.shape[0], self.txt_groups, -1)
            t_txt = torch.einsum("bgk,gkd->bgd", g, self.txt_w) + self.txt_b
        cls = self.cls.expand(num.shape[0], -1, -1)
        parts = [cls, t_num, t_cat, t_txt]
        if words is not None and self.word_len:
            words = words.long()
            parts.append(self.word_emb(words) + self.word_pos)
        return torch.cat(parts, dim=1)

    def attn_mask(self, words=None):
        """Маска attention (b, 1, 1, T): True — токен виден; None, если словесных токенов нет."""
        if words is None or not self.word_len:
            return None
        b = words.shape[0]
        head = torch.ones(b, self.n_tokens - self.word_len, dtype=torch.bool, device=words.device)
        return torch.cat([head, words != 0], dim=1)[:, None, None, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dropout = n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, attn_mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class Block(nn.Module):
    def __init__(self, d, n_heads, d_ffn, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d_ffn), ReGLU(), nn.Dropout(ffn_dropout), nn.Linear(d_ffn, d))
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask))
        return x + self.drop(self.ffn(self.ln2(x)))


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: list[int], txt_dim: int, hp: dict | None = None,
                 txt_groups: int = 1, word_vocab: int = 0, word_len: int = 0):
        super().__init__()
        hp = {**DEFAULT_HP, **(hp or {})}
        d = hp["d_token"]
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, txt_dim, d, txt_groups, word_vocab, word_len)
        self.blocks = nn.ModuleList([
            Block(d, hp["n_heads"], hp["d_ffn"], hp["attn_dropout"], hp["ffn_dropout"], hp["residual_dropout"])
            for _ in range(hp["n_blocks"])
        ])
        self.head_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def embed(self, num, num_mask, cat, txt, words=None):
        """Нормализованный выход [CLS] — эмбеддинг объекта (используется в ПР1 для MMD)."""
        x = self.tokenizer(num, num_mask, cat, txt, words)
        mask = self.tokenizer.attn_mask(words)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head_ln(x[:, 0])

    def forward(self, num, num_mask, cat, txt, words=None):
        return self.head(F.relu(self.embed(num, num_mask, cat, txt, words))).squeeze(-1)
