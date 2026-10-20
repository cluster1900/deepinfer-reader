from minidi.blocks import BlockManager
from minidi.scheduler import Scheduler
from minidi.sequence import Sequence, Status


def mk(i, n, out=0, bs=4):
    s = Sequence(i, list(range(n)), block_size=bs, max_tokens=100)
    s.tokens += [7] * out
    return s


def test_prompt_and_decode_never_mixed():
    bm = BlockManager(64)
    sch = Scheduler(bm, max_num_batched_tokens=16)
    for i in range(3):
        sch.add(mk(i, 5))
    o = sch.schedule()
    assert o.prompt_run and [s.seq_id for s in o.scheduled] == [0, 1, 2]
    assert sch.running == []                     # prefill 完成前不进 running
    sch.process_outputs(o, [[1], [1], [1]])
    assert [s.seq_id for s in sch.running] == [0, 1, 2]
    o = sch.schedule()                           # waiting 空了 → decode 步
    assert not o.prompt_run and o.seqlens_q == [1, 1, 1]


def test_chunked_prefill_stops_after_first_chunk_and_requeues_front():
    bm = BlockManager(64)
    sch = Scheduler(bm, max_num_batched_tokens=8)
    a, b = mk(0, 3), mk(1, 20)
    sch.add(a); sch.add(b)
    o = sch.schedule()
    assert [s.seq_id for s in o.scheduled] == [0, 1] and o.seqlens_q == [3, 5]
    sch.process_outputs(o, [[9], []])
    assert sch.waiting[0] is b and b.num_cached == 5 and b.status == Status.WAITING


def test_no_chunk_when_disabled():
    bm = BlockManager(64)
    sch = Scheduler(bm, max_num_batched_tokens=8, max_model_len=8, chunked_prefill=False)
    sch.add(mk(0, 3)); sch.add(mk(1, 6))
    o = sch.schedule()
    assert o.seqlens_q == [3]


def test_too_long_prompt_ignored():
    sch = Scheduler(BlockManager(64), max_model_len=10)
    sch.add(mk(0, 11))
    o = sch.schedule()
    assert o.ignored[0].status == Status.FINISHED_IGNORED


def test_lcfs_preempts_longest_output_first():
    """running 按 output_len 降序，块不够时从头部（输出最长）开始抢占 —— 与 scheduler.hpp L622-L644 相同。"""
    bm = BlockManager(7)                         # 6 个可用块
    sch = Scheduler(bm, max_num_seqs=8)
    seqs = [mk(0, 4, out=4), mk(1, 4, out=0), mk(2, 4, out=8)]  # 长度 8、4、12，正好占满整块
    for s in seqs:
        s.block_table = bm.allocate((s.get_len() - 1) // 4 + 1)
        s.num_cached = s.get_len() - 1
        s.status = Status.RUNNING
    # 现在 seq0 2 块、seq1 1 块、seq2 3 块，0 块空闲；下一个 token 都需要新块
    sch.running = list(seqs)
    sch.decode_step = 1
    for s in seqs:
        s.tokens.append(1)
    o = sch.schedule()
    assert not o.prompt_run
    assert sch.log[0] == "preempt-recompute seq2"          # 输出最长的先被抢
    assert {s.seq_id for s in o.scheduled} == {0, 1}


def test_stop_preempted_request():
    bm = BlockManager(3)
    sch = Scheduler(bm, stop_preempted_request=True)
    a, b = mk(0, 4, out=1), mk(1, 4, out=0)
    for s in (a, b):
        s.block_table = bm.allocate(1)
        s.status = Status.RUNNING
    sch.running = [a, b]
    sch.decode_step = 1
    o = sch.schedule()
    assert a.status == Status.FINISHED_PREEMPTED and o.ignored == [a]


def test_check_stop_eos_and_max_tokens():
    sch = Scheduler(BlockManager(8), eos_id=3)
    s = mk(0, 2)
    s.max_tokens = 2
    sch._check_stop(s, 5)
    assert not s.is_finished()
    s.tokens.append(3)
    sch._check_stop(s, 3)
    assert s.status == Status.FINISHED_STOPPED


def test_append_slot_frees_redundant_blocks_after_draft_rejection():
    """对应 scheduler.hpp L821-L825：上一步因 draft 预留了第 2 块，draft 被拒且本步不再需要时自动归还。"""
    bm = BlockManager(6)
    sch = Scheduler(bm, max_num_seqs=8)
    s = mk(0, 3, out=0, bs=4)
    # 模拟上一步有 3 个 draft（3 + 3 = 6 token -> 2 块），但最后只接受了 0 个 draft（只多 1 个主 token，共 4 token -> 1 块）
    s.block_table = bm.allocate(2)
    s.tokens.append(9)                           # 长度 4，draft 清空后 get_num_blocks() == 1
    s.num_cached = 3
    s.status = Status.RUNNING
    sch.running = [s]
    sch.decode_step = 1
    free_before = bm.num_free()
    o = sch.schedule()
    assert not o.prompt_run and len(s.block_table) == 1
    assert bm.num_free() == free_before + 1

