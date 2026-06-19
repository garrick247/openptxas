"""Control-flow graph + GPR liveness reconstructed from a final SASS instruction
stream (list of SassInstr).  Foundation for CFG-aware scheduling: lets the
scheduler reason about cross-block liveness (forward edges, back-edges, merges)
instead of the conservative "regs used anywhere outside the run" approximation.

A block is a maximal straight-line run of instructions with a single entry.
Leaders: index 0, any instruction tagged with a `// label:` comment (block entry
emitted by isel), and any instruction following a BRA/EXIT.  Successors come from
the block's terminator: EXIT -> none; unconditional BRA -> [target]; conditional
(@P) BRA -> [target, fall-through]; no branch -> [fall-through].
"""
from __future__ import annotations
from sass.scoreboard import _get_opcode, _get_src_regs, _get_dest_regs

_BRA = 0x947
_EXIT = 0x94d
_NOP = 0x918
_PT = 0x7  # predicate-true guard nibble


def _is_label_leader(comment):
    # isel tags the first instr of a labeled block as "// <label>: <rest>"
    if not comment.startswith("// "):
        return None
    head = comment[3:]
    if ": " not in head and not head.endswith(":"):
        return None
    lbl = head.split(":", 1)[0].strip()
    # labels look like .L_x_NN or accumulate_numerators_else_25 etc. (no spaces)
    return lbl if lbl and " " not in lbl else None


def _bra_target(comment):
    if "BRA " not in comment:
        return None
    tok = comment.split("BRA ", 1)[1].split()[0].strip()
    return tok or None


def _is_conditional(raw):
    # predicated (guard != PT) BRA falls through when the guard is false
    return ((raw[1] >> 4) & 0xF) != _PT


def build_cfg(instrs):
    """Return dict: blocks=[(start,end)], label2blk={lbl:blkidx},
    succ=[[blkidx,...]], pred=[[blkidx,...]], blk_of=[blkidx per instr]."""
    n = len(instrs)
    if n == 0:
        return {"blocks": [], "label2blk": {}, "succ": [], "pred": [], "blk_of": []}
    leader = [False] * n
    leader[0] = True
    label_at = {}  # instr idx -> label
    for i, si in enumerate(instrs):
        lbl = _is_label_leader(si.comment)
        if lbl is not None:
            leader[i] = True
            label_at[i] = lbl
        op = _get_opcode(si.raw)
        if op in (_BRA, _EXIT) and i + 1 < n:
            leader[i + 1] = True
    starts = [i for i in range(n) if leader[i]]
    blocks = [(starts[k], starts[k + 1] if k + 1 < len(starts) else n)
              for k in range(len(starts))]
    blk_of = [0] * n
    for b, (s, e) in enumerate(blocks):
        for i in range(s, e):
            blk_of[i] = b
    label2blk = {label_at[s]: b for b, (s, e) in enumerate(blocks) if s in label_at}

    succ = [[] for _ in blocks]
    for b, (s, e) in enumerate(blocks):
        # last non-NOP instruction = terminator
        term = None
        for i in range(e - 1, s - 1, -1):
            if _get_opcode(instrs[i].raw) != _NOP:
                term = i
                break
        fall = b + 1 if b + 1 < len(blocks) else None
        if term is None:
            if fall is not None:
                succ[b].append(fall)
            continue
        op = _get_opcode(instrs[term].raw)
        if op == _EXIT:
            pass  # no successors
        elif op == _BRA:
            tgt_lbl = _bra_target(instrs[term].comment)
            tgt = label2blk.get(tgt_lbl)
            if tgt is not None:
                succ[b].append(tgt)
            if _is_conditional(instrs[term].raw) and fall is not None:
                succ[b].append(fall)
        else:
            if fall is not None:
                succ[b].append(fall)
    pred = [[] for _ in blocks]
    for b, ss in enumerate(succ):
        for t in ss:
            pred[t].append(b)
    return {"blocks": blocks, "label2blk": label2blk, "succ": succ,
            "pred": pred, "blk_of": blk_of}


def _block_def_use(instrs, s, e):
    """GPR def/use for liveness. use = read before any write in the block;
    def = written (unconditionally). Predicated writes are treated as use+def
    (they read the old value when the guard is false) so the reg stays live-in."""
    use = set()
    defd = set()
    for i in range(s, e):
        raw = instrs[i].raw
        if _get_opcode(raw) == _NOP:
            continue
        for r in _get_src_regs(raw):
            if r < 255 and r not in defd:
                use.add(r)
        predicated = ((raw[1] >> 4) & 0xF) != _PT
        for r in _get_dest_regs(raw):
            if r >= 255:
                continue
            if predicated:
                if r not in defd:
                    use.add(r)  # conditional write reads old value
            else:
                defd.add(r)
    return use, defd


def compute_liveness(instrs, cfg):
    """Backward dataflow to fixpoint. Returns (live_in, live_out) lists of GPR
    sets per block."""
    nb = len(cfg["blocks"])
    use = [None] * nb
    defd = [None] * nb
    for b, (s, e) in enumerate(cfg["blocks"]):
        use[b], defd[b] = _block_def_use(instrs, s, e)
    live_in = [set() for _ in range(nb)]
    live_out = [set() for _ in range(nb)]
    succ = cfg["succ"]
    changed = True
    while changed:
        changed = False
        for b in range(nb - 1, -1, -1):
            lo = set()
            for t in succ[b]:
                lo |= live_in[t]
            li = use[b] | (lo - defd[b])
            if lo != live_out[b] or li != live_in[b]:
                live_out[b] = lo
                live_in[b] = li
                changed = True
    return live_in, live_out


def live_after_index(instrs, cfg, live_out, idx):
    """GPRs live immediately AFTER instruction `idx` (true CFG liveness, incl.
    back-edges). Recompute backward from the block's live_out to `idx`."""
    b = cfg["blk_of"][idx]
    s, e = cfg["blocks"][b]
    live = set(live_out[b])
    for i in range(e - 1, idx, -1):
        raw = instrs[i].raw
        if _get_opcode(raw) == _NOP:
            continue
        predicated = ((raw[1] >> 4) & 0xF) != _PT
        if not predicated:
            for r in _get_dest_regs(raw):
                if r < 255:
                    live.discard(r)
        for r in _get_src_regs(raw):
            if r < 255:
                live.add(r)
    return live
