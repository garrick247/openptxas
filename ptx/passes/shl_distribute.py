"""
Offset-through-shift distribution for FORGE per-limb address chains.

ptxas 13.3 lowers `base + (idx*2^S1 + k)*2^S2` (the per-limb effective
address FORGE emits, with k a small constant differing per limb) to a
SINGLE shared base address `base + idx*2^(S1+S2)` plus a constant memory
offset `k*2^S2` folded into each LDG/STG.  openptxas instead computes a
full address per limb, which downstream emits one LEA pair (or IMAD.WIDE)
per limb instead of one shared address + cheap offset loads.

This pass performs the algebraically-exact rewrite at the PTX-IR level:

    shl.b64  %M1, %X, S1          (M1 = idx << S1; may be multi-used, kept)
    add.u64  %A,  %M1, K          (A single-use; K a small u64 immediate)
    shl.b64  %M2, %A,  S2         (M2 single-use)
    add.u64  %F,  %B,  %M2        (F single-use, only as a global ld/st base)
    ld/st.global [%F], ...

  =>

    shl.b64  %Mn, %X, (S1+S2)     (fresh; idx << (S1+S2))
    add.u64  %F,  %B, %Mn         (consumer rebased onto %Mn)
    ld/st.global [%F + (K << S2)], ...

  using the identity  (idx*2^S1 + K) * 2^S2 == idx*2^(S1+S2) + K*2^S2,
  which holds exactly in modular 2^64 arithmetic (shifts and adds wrap
  consistently).  The constant K*2^S2 is folded straight into the memory
  operand offset; `%X << (S1+S2)` then feeds the UR-base LEA selection
  (idx_hi_zero path) the same way `mul_distribute` feeds IMAD.WIDE.

Cross-limb sharing of `%B + %X<<(S1+S2)` is left to the existing CSE
passes (cvt_shl_cse / common_mul_sum / load_cse), exactly as mul_distribute
relies on downstream folding.

# Conservative gating (correctness)
- %A single-def AND single-use (only by the second shl).
- %M2 single-def AND single-use (only by the consumer add).
- %F single-def AND single-use, and that use is a global ld/st MemOp base.
- %M1's defining shl is unpredicated (we read %X unconditionally).
- The chain instructions share one predicate; new instrs inherit it.
- K << S2 must fit the 24-bit signed LDG.E/STG.E offset range.
- S1+S2 in [1,63]; the combined shift must not be a no-op.

Toggle: this pass is OFF unless OPENPTXAS_ENABLE_SHL_DISTRIBUTE is set
(staged building block — see project_openptxas.md; needs kraken runtime
validation before it becomes default).
"""
from __future__ import annotations

import os
from typing import Optional

from ..ir import Function, ImmOp, Instruction, MemOp, RegOp, VectorRegOp, RegDecl, U64


_INT64 = ("u64", "s64", "b64")
_OFF_MAX = (1 << 23) - 1
_OFF_MIN = -(1 << 23)


def _is_i64(inst: Instruction) -> bool:
    return bool(inst.types) and any(t in _INT64 for t in inst.types)


def _walk(fn: Function):
    def_count: dict[str, int] = {}
    use_count: dict[str, int] = {}
    def_instr: dict[str, Instruction] = {}
    global_base: set[str] = set()
    for bb in fn.blocks:
        for inst in bb.instructions:
            d = inst.dest
            if isinstance(d, VectorRegOp):
                for r in (d.regs or ()):
                    def_count[r] = def_count.get(r, 0) + 1
                    def_instr[r] = inst
            elif isinstance(d, RegOp):
                def_count[d.name] = def_count.get(d.name, 0) + 1
                def_instr[d.name] = inst
            for src in (inst.srcs or []):
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    use_count[src.name] = use_count.get(src.name, 0) + 1
                elif isinstance(src, MemOp):
                    b = src.base
                    if isinstance(b, str) and b:
                        # MemOp.base may be stored with or without the % sigil;
                        # normalise to the %-form used by RegOp.name.
                        bn = b if b.startswith('%') else f'%{b}'
                        use_count[bn] = use_count.get(bn, 0) + 1
                        if (inst.op in ('ld', 'st', 'atom')
                                and 'global' in (inst.types or ())):
                            global_base.add(bn)
    return def_count, use_count, def_instr, global_base


def _shl_of(inst: Instruction):
    """If inst is `shl.b64 %D, %X, S_imm`, return (D_name, X_reg, S). Else None."""
    if inst is None or inst.op != "shl" or not _is_i64(inst):
        return None
    if inst.mods:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    x, s = inst.srcs[0], inst.srcs[1]
    if not isinstance(x, RegOp) or isinstance(x, VectorRegOp):
        return None
    if not isinstance(s, ImmOp):
        return None
    return (inst.dest.name, x, s.value)


def _alloc_vreg(fn: Function) -> str:
    if not hasattr(fn, "_shl_distribute_next_id"):
        fn._shl_distribute_next_id = 0
    while True:
        n = fn._shl_distribute_next_id
        fn._shl_distribute_next_id += 1
        cand = f"%shldist_{n}"
        if not any(rd.names and cand in rd.names for rd in fn.reg_decls):
            fn.reg_decls.append(RegDecl(type=U64, name=cand.lstrip('%'), count=1))
            return cand


def run_function(fn: Function) -> int:
    if not os.environ.get("OPENPTXAS_ENABLE_SHL_DISTRIBUTE"):
        return 0
    def_count, use_count, def_instr, global_base = _walk(fn)
    n = 0
    for bb in fn.blocks:
        instrs = bb.instructions
        i = 0
        while i < len(instrs):
            # Anchor on the leading add: `[@p] add.u64 %A, %M1, K`.
            cand = instrs[i]
            if cand.op != "add" or not _is_i64(cand) or cand.mods:
                i += 1
                continue
            if not isinstance(cand.dest, RegOp) or isinstance(cand.dest, VectorRegOp):
                i += 1
                continue
            if len(cand.srcs or []) != 2:
                i += 1
                continue
            a_name = cand.dest.name
            if def_count.get(a_name, 0) != 1 or use_count.get(a_name, 0) != 1:
                i += 1
                continue
            # Identify %M1 (a shl.b64 result) and K (small u64 immediate).
            m1_op = k_imm = None
            for src in cand.srcs:
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    df = def_instr.get(src.name)
                    if df is not None and _shl_of(df) is not None:
                        m1_op = src
            if m1_op is None:
                i += 1
                continue
            other = cand.srcs[0] if cand.srcs[1] is m1_op else cand.srcs[1]
            if not isinstance(other, ImmOp):
                i += 1
                continue
            K = other.value & 0xFFFFFFFFFFFFFFFF
            m1 = _shl_of(def_instr[m1_op.name])
            if m1 is None:
                i += 1
                continue
            _m1name, x_reg, S1 = m1
            if def_instr[m1_op.name].pred is not None:
                i += 1
                continue
            # Second shl: `[@p] shl.b64 %M2, %A, S2` must follow.
            if i + 1 >= len(instrs):
                i += 1
                continue
            sh2 = _shl_of(instrs[i + 1])
            if sh2 is None or instrs[i + 1].srcs[0].name != a_name:
                i += 1
                continue
            m2_name, _a2, S2 = sh2
            if (instrs[i + 1].pred, instrs[i + 1].neg) != (cand.pred, cand.neg):
                i += 1
                continue
            if def_count.get(m2_name, 0) != 1 or use_count.get(m2_name, 0) != 1:
                i += 1
                continue
            # Consumer add: `[@p] add.u64 %F, %B, %M2`.
            if i + 2 >= len(instrs):
                i += 1
                continue
            cadd = instrs[i + 2]
            if cadd.op != "add" or not _is_i64(cadd) or cadd.mods:
                i += 1
                continue
            if (cadd.pred, cadd.neg) != (cand.pred, cand.neg):
                i += 1
                continue
            if len(cadd.srcs or []) != 2 or not isinstance(cadd.dest, RegOp):
                i += 1
                continue
            ca, cb = cadd.srcs[0], cadd.srcs[1]
            if isinstance(ca, RegOp) and ca.name == m2_name and isinstance(cb, RegOp):
                b_reg = cb
            elif isinstance(cb, RegOp) and cb.name == m2_name and isinstance(ca, RegOp):
                b_reg = ca
            else:
                i += 1
                continue
            f_name = cadd.dest.name
            if def_count.get(f_name, 0) != 1 or use_count.get(f_name, 0) != 1:
                i += 1
                continue
            if f_name not in global_base:
                i += 1
                continue
            # Offset = K << S2; must fit the signed 24-bit memop offset.
            S = S1 + S2
            if S < 1 or S > 63 or (1 << S) >> 32 != 0:
                i += 1
                continue
            off = (K << S2) & 0xFFFFFFFFFFFFFFFF
            soff = off - (1 << 64) if off >= (1 << 63) else off
            if not (_OFF_MIN <= soff <= _OFF_MAX):
                i += 1
                continue
            # --- Rewrite -----------------------------------------------------
            # new shl: %Mn = %X << S    (S = S1+S2)
            mn = _alloc_vreg(fn)
            new_shl = Instruction(op="shl", types=["b64"],
                                  dest=RegOp(mn),
                                  srcs=[RegOp(x_reg.name), ImmOp(S)],
                                  pred=cand.pred, neg=cand.neg)
            # consumer add rebased: %F = %B + %Mn
            cadd.srcs = [RegOp(b_reg.name), RegOp(mn)]
            # fold K<<S2 into the memory op that bases on %F
            for inst in instrs:
                for si, src in enumerate(inst.srcs or []):
                    if (isinstance(src, MemOp) and isinstance(src.base, str)
                            and (src.base == f_name
                                 or src.base == f_name.lstrip('%')
                                 or f'%{src.base}' == f_name)):
                        inst.srcs[si] = MemOp(base=src.base, offset=src.offset + soff)
            # Replace the leading add + second shl with the single new shl;
            # the original first shl (%M1) is preserved (often multi-used).
            instrs[i] = new_shl          # was: add %A,%M1,K
            del instrs[i + 1]            # was: shl %M2,%A,S2
            n += 1
            i += 1
    return n


def run(mod) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
