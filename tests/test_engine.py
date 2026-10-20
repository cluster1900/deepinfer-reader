import numpy as np
import pytest

from minidi.engine import Engine, reference_generate
from minidi.model import TinyLM
from minidi.zos import ZOSEngine, simulate_timeline

M = TinyLM()


def prompts(seed, k=8, hi=30):
    rng = np.random.default_rng(seed)
    return [list(map(int, rng.integers(0, 64, int(n)))) for n in rng.integers(1, hi, k)]


def ref(ps, n, eos=None):
    return [reference_generate(M, p, n, eos) for p in ps]


@pytest.mark.parametrize("seed", range(12))
def test_engine_matches_reference_with_preemption(seed):
    rng = np.random.default_rng(100 + seed)
    nb = int(rng.integers(14, 40))
    eos = int(rng.integers(0, 64))
    ps = prompts(seed)
    e = Engine(M, num_blocks=nb, max_num_seqs=6, max_num_batched_tokens=16, max_model_len=40, eos_id=eos)
    assert e.generate(ps, max_tokens=10) == ref(ps, 10, eos)
    assert e.bm.num_free() == nb - 1


def test_preemption_actually_happens():
    total = 0
    for seed in range(12):
        e = Engine(M, num_blocks=14, max_num_seqs=6, max_num_batched_tokens=16, max_model_len=40)
        e.generate(prompts(seed), max_tokens=10)
        total += e.scheduler.num_preempted
    assert total > 0


def test_kv_capacity_guard():
    with pytest.raises(ValueError):
        Engine(M, num_blocks=10, block_size=4, max_model_len=40)


@pytest.mark.parametrize("noise", [0.0, 0.2, 0.6])
def test_speculative_decoding_is_lossless(noise):
    ps = prompts(3, k=5)
    e = Engine(M, num_blocks=80, max_num_batched_tokens=32, draft=TinyLM(noise=noise), nextn=3)
    assert e.generate(ps, max_tokens=12) == ref(ps, 12)
    if noise == 0.0:
        assert e.num_accepted == e.num_draft      # 草稿就是目标模型：全部接受


def test_seeded_sampling_is_batch_invariant():
    ps = prompts(5, k=6)
    a = Engine(M, num_blocks=80, max_num_batched_tokens=64).generate(ps, max_tokens=8, temperature=1.0, seed=7)
    b = Engine(M, num_blocks=80, max_num_batched_tokens=8, max_num_seqs=2).generate(ps[::-1], max_tokens=8,
                                                                                    temperature=1.0, seed=7)
    assert a == b[::-1]


@pytest.mark.parametrize("seed", range(20))
def test_zos_matches_sync(seed):
    rng = np.random.default_rng(seed)
    nb = int(rng.integers(14, 40))
    kw = dict(num_blocks=nb, max_num_seqs=6, max_num_batched_tokens=16, max_model_len=40, eos_id=int(rng.integers(0, 64)))
    ps = prompts(seed)
    z = ZOSEngine(M, **kw)
    assert z.generate(ps, max_tokens=10) == Engine(M, **kw).generate(ps, max_tokens=10)
    assert z.bm.num_free() == nb - 1


def test_timeline_zos_hides_cpu():
    _, t_sync = simulate_timeline(10, zos=False)
    _, t_zos = simulate_timeline(10, zos=True)
    # 同步：每步 sched+launch+fwd+samp+post = 12；ZOS 稳态每步 = max(GPU 7, CPU 5) = 7
    assert t_sync == 120.0
    assert t_zos < t_sync and t_zos == 3 + 7 * 10 + 2
