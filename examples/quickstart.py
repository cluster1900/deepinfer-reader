"""minidi 快速体验：同一批请求分别用同步引擎、异步引擎（ZOS）、投机解码跑一遍，
检查输出与无缓存参考实现逐 token 一致，并打印调度统计。

    python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from minidi.engine import Engine, reference_generate  # noqa: E402
from minidi.model import TinyLM  # noqa: E402
from minidi.zos import ZOSEngine  # noqa: E402


def main() -> None:
    model = TinyLM()
    rng = np.random.default_rng(0)
    prompts = [list(map(int, rng.integers(0, 64, int(n)))) for n in rng.integers(3, 30, 8)]
    ref = [reference_generate(model, p, 12) for p in prompts]
    kw = dict(num_blocks=24, max_num_seqs=6, max_num_batched_tokens=16, max_model_len=60)

    e = Engine(model, **kw)
    out = e.generate(prompts, max_tokens=12)
    n_prefill = sum(1 for s in e.steps if s[0])
    print(f"[同步]   一致={out == ref}  steps={len(e.steps)}  prefill 步={n_prefill}  抢占={e.scheduler.num_preempted}")

    z = ZOSEngine(model, **kw)
    print(f"[ZOS]    一致={z.generate(prompts, max_tokens=12) == ref}  steps={len(z.steps)}")

    s = Engine(model, draft=TinyLM(noise=0.3), nextn=3, **kw)
    out = s.generate(prompts, max_tokens=12)
    rate = s.num_accepted / max(1, s.num_draft)
    print(f"[投机]   一致={out == ref}  steps={len(s.steps)}  draft 接受率={rate:.2f}")


if __name__ == "__main__":
    main()
