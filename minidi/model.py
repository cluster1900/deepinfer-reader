"""TinyLM：单层单头注意力的玩具语言模型，带分页 KV Cache。

对照 DeepInfer：
- 真实模型在 deepinfer/model_executor/models/deepseek.py（DeepseekForCausalLM），
  注意力在 v4_modules/attn.py，KV 写入在 tilelang_ops/store_kv_cache.py。
- 这里的 paged_forward 接收"扁平变长"批次（所有请求的 query token 拼在一起，
  用 seqlens_q / seqlens_k / block_tables 描述），与 DeepInfer 的 InputMetadata 思路一致。
权重固定随机种子，float64 计算，便于和不带缓存的参考实现逐位比较。
"""
from __future__ import annotations

import numpy as np


class TinyLM:
    def __init__(self, vocab: int = 64, dim: int = 32, max_len: int = 512, seed: int = 0, noise: float = 0.0):
        rng = np.random.default_rng(seed)
        self.vocab, self.dim, self.max_len = vocab, dim, max_len
        s = 1.0 / np.sqrt(dim)
        self.emb = rng.normal(0, 1, (vocab, dim))
        self.wq = rng.normal(0, s, (dim, dim))
        self.wk = rng.normal(0, s, (dim, dim))
        self.wv = rng.normal(0, s, (dim, dim))
        self.wo = rng.normal(0, s, (dim, vocab))
        pos = np.arange(max_len)[:, None]
        freq = np.exp(-np.log(10000.0) * (np.arange(0, dim, 2) / dim))
        self.pos = np.zeros((max_len, dim))
        self.pos[:, 0::2] = np.sin(pos * freq)
        self.pos[:, 1::2] = np.cos(pos * freq)
        if noise > 0:  # 草稿模型：同一个模型加噪声，接受率介于 0 和 1 之间
            nrng = np.random.default_rng(seed + 1000)
            for name in ("wq", "wk", "wv", "wo"):
                w = getattr(self, name)
                setattr(self, name, w + nrng.normal(0, noise * s, w.shape))

    # ---------- 参考实现：每次从头算，没有缓存 ----------
    def reference_logits(self, tokens: list[int]) -> np.ndarray:
        """返回 [len(tokens), vocab]，第 i 行是看完 tokens[:i+1] 后的下一个 token 分布。"""
        n = len(tokens)
        h = self.emb[tokens] + self.pos[:n]
        q, k, v = h @ self.wq, h @ self.wk, h @ self.wv
        att = q @ k.T / np.sqrt(self.dim)
        att = np.where(np.tril(np.ones((n, n), bool)), att, -np.inf)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        return (h + att @ v) @ self.wo

    # ---------- 分页实现 ----------
    def paged_forward(self, kv: "PagedKV", batch: list[tuple[list[int], int, list[int]]]) -> np.ndarray:
        """batch 中每项为 (query_tokens, start_pos, block_table)。

        query_tokens 写入位置 [start_pos, start_pos+len)，并对 [0, start_pos+len) 做因果注意力。
        返回扁平 logits [sum(len(query_tokens)), vocab]。
        """
        out = []
        for q_tokens, start, table in batch:
            n = len(q_tokens)
            pos = np.arange(start, start + n)
            h = self.emb[q_tokens] + self.pos[pos]
            q, k, v = h @ self.wq, h @ self.wk, h @ self.wv
            kv.write(table, pos, k, v)                      # 先写 KV（对应 store_kv_cache）
            K, V = kv.read(table, start + n)                # 再按 block_table 读回整段上下文
            att = q @ K.T / np.sqrt(self.dim)
            mask = np.arange(start + n)[None, :] <= pos[:, None]
            att = np.where(mask, att, -np.inf)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            out.append((h + att @ V) @ self.wo)
        return np.concatenate(out) if out else np.zeros((0, self.vocab))


class PagedKV:
    """物理 KV 池：[num_blocks, block_size, dim]。块 0 保留不用，与 DeepInfer Blocks 的约定一致。"""

    def __init__(self, num_blocks: int, block_size: int, dim: int):
        self.block_size = block_size
        self.k = np.zeros((num_blocks, block_size, dim))
        self.v = np.zeros((num_blocks, block_size, dim))

    def _slots(self, table: list[int], pos: np.ndarray):
        blocks = np.asarray(table)[pos // self.block_size]
        return blocks, pos % self.block_size

    def write(self, table, pos, k, v):
        b, o = self._slots(table, pos)
        assert (b > 0).all(), "写到了保留块 0：block_table 不够长"
        self.k[b, o] = k
        self.v[b, o] = v

    def read(self, table, length):
        pos = np.arange(length)
        b, o = self._slots(table, pos)
        return self.k[b, o], self.v[b, o]
