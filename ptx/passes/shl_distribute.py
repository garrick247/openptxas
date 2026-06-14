"""
Per-limb address coalescing for FORGE store/gather kernels (matches the
shape ptxas 13.3 emits).

FORGE emits, per limb k, a full effective-address chain
    [ add %A,%M1,k ; ]  shl %M2,%A,S2 ; add %F,%B,%M2 ; ld/st [%F]
where %M1 = X<<S1 is a shared intermediate index, %B a base pointer, and k
a small per-limb constant.  ptxas keeps %M1 as the LEA index, scales by the
OUTER shift S2, and collapses all limbs sharing (%M1, S2, %B) to ONE base
address  %B + %M1<<S2  plus a folded per-limb memory offset  k<<S2.

This pass reproduces that structure at the PTX-IR level, using
  (X*2^S1 + k) * 2^S2 == (X*2^S1)*2^S2 + k*2^S2   (exact mod 2^64).
For the first limb of each (%M1,S2,%B,pred) group it emits, into FRESH
vregs (so the shared base is never clobbered by FORGE's vreg reuse):
    shl %Mn,%M1,S2 ; add %Fsh,%B,%Mn ; ld/st [%Fsh + k<<S2]
and every later limb in the group reuses %Fsh:  ld/st [%Fsh + k<<S2],
dropping its now-dead add/shl/add chain.  The kept index is %M1 (= X<<S1,
NON-zero-extended) — exactly what ptxas feeds to the non-hi-zero UR-base
LEA (see isel _emit_imad_wide_fused, src_c = idx.hi).

# Block-local liveness (handles FORGE's heavily-reused vreg names)
All gating is per basic block over instruction windows, NOT function-wide
def/use counts (which reject reused names).  %M2's sole reader is the
consumer add; %F's sole reader is one global ld/st base; for k!=0 %A's sole
reader is the outer shift (its leading add is then dead); %M1 and %B must be
unwritten between a group's anchor and each reuse (same value).  k<<S2 must
fit the signed 24-bit LDG.E/STG.E offset; S2 in [1,4] (LEA scale range).

GPU-validated (2026-06-14): with OPENPTXAS_ENABLE_SHL_DISTRIBUTE alone,
bit_reverse_qm31 is byte/behaviour-identical to ptxas on the 5090 at
N=256 AND N=4096 (/tmp/driver.py).  Flag-off keeps the whole reference set
byte-identical to the prior commit.

Toggle: OFF unless OPENPTXAS_ENABLE_SHL_DISTRIBUTE is set.  Pairs with the
non-hi-zero LEA selection (OPENPTXAS_ENABLE_NONHZ_LEA) which consumes the
%M1 index this pass exposes — but that LEA path is NOT yet correct (it
reads idx.hi via ctx.ra.hi but the intermediate's high half is not
materialised; crashes at N=4096).  This coalescing pass is correct on its
own; the LEA emission is the open item.
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


def _writes(inst: Instruction, name: str) -> bool:
    d = inst.dest
    if isinstance(d, VectorRegOp):
        return name in (d.regs or ())
    return isinstance(d, RegOp) and d.name == name


def _shl_of(inst: Instruction):
    """`shl.b64 %D, %X, S_imm` -> (D_name, X_RegOp, S).  Else None."""
    if inst is None or inst.op != "shl" or not _is_i64(inst) or inst.mods:
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


def _add_imm_of(inst: Instruction):
    """`add.u64 %A, %M1, K_imm` -> (A_name, M1_RegOp, K).  Either src order."""
    if inst is None or inst.op != "add" or not _is_i64(inst) or inst.mods:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    a, b = inst.srcs
    if isinstance(a, RegOp) and not isinstance(a, VectorRegOp) and isinstance(b, ImmOp):
        return (inst.dest.name, a, b.value)
    if isinstance(b, RegOp) and not isinstance(b, VectorRegOp) and isinstance(a, ImmOp):
        return (inst.dest.name, b, a.value)
    return None


def _last_write_before(instrs, idx, name) -> Optional[int]:
    for j in range(idx - 1, -1, -1):
        if _writes(instrs[j], name):
            return j
    return None


def _next_write_after(instrs, idx, name) -> int:
    for j in range(idx + 1, len(instrs)):
        if _writes(instrs[j], name):
            return j
    return len(instrs)


def _writes_between(instrs, lo, hi, name) -> bool:
    """True if any instruction in (lo, hi) writes `name`."""
    for j in range(lo + 1, hi):
        if _writes(instrs[j], name):
            return True
    return False


def _reg_readers(instrs, lo, hi, name):
    """Positions in [lo,hi) reading `name` as a RegOp source."""
    out = []
    for j in range(lo, hi):
        for s in (instrs[j].srcs or []):
            if isinstance(s, RegOp) and not isinstance(s, VectorRegOp) and s.name == name:
                out.append(j)
                break
    return out


def _membase_readers(instrs, lo, hi, name):
    """Positions in [lo,hi) using `name` as a MemOp base (sigil-normalised)."""
    out = []
    for j in range(lo, hi):
        for s in (instrs[j].srcs or []):
            if isinstance(s, MemOp) and isinstance(s.base, str):
                bn = s.base if s.base.startswith('%') else f'%{s.base}'
                if bn == name:
                    out.append(j)
                    break
    return out


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
    n = 0
    for bb in fn.blocks:
        instrs = bb.instructions
        cse: dict = {}        # (M1, S2, B, pred) -> (shared_base_vreg, anchor_idx)
        dead: set = set()     # id() of instructions to drop after the block walk
        i = 0
        while i < len(instrs):
            sh2 = _shl_of(instrs[i])             # outer shift: %M2 = %A << S2
            if sh2 is None:
                i += 1
                continue
            m2_name, a_op, S2 = sh2
            a_name = a_op.name
            pred = (instrs[i].pred, instrs[i].neg)

            a_def_idx = _last_write_before(instrs, i, a_name)
            if a_def_idx is None:
                i += 1
                continue
            a_def = instrs[a_def_idx]

            # Resolve (X, S1, K, m1_name, is_k0).
            shl_a = _shl_of(a_def)
            add_a = _add_imm_of(a_def)
            if shl_a is not None:
                # k == 0 slot: %A is itself idx<<S1.  %A/%M1 may be shared.
                m1_name, x_op, S1 = shl_a
                K = 0
                is_k0 = True
            elif add_a is not None:
                # k != 0: %A = %M1 + K ; %M1 must be defined by a shift.
                _aname, m1_op, K = add_a
                m1_name = m1_op.name
                m1_def_idx = _last_write_before(instrs, a_def_idx, m1_name)
                if m1_def_idx is None:
                    i += 1
                    continue
                shl_m1 = _shl_of(instrs[m1_def_idx])
                if shl_m1 is None:
                    i += 1
                    continue
                _m1n, x_op, S1 = shl_m1
                is_k0 = False
                # %A's only reader in its live range must be this outer shift.
                a_end = _next_write_after(instrs, a_def_idx, a_name)
                if _reg_readers(instrs, a_def_idx + 1, a_end, a_name) != [i]:
                    i += 1
                    continue
            else:
                i += 1
                continue

            if S2 < 1 or S2 > 4:          # LEA encodes scale 0..4 (the OUTER shift)
                i += 1
                continue

            # Consumer add: the unique reader of %M2 in its window.
            m2_end = _next_write_after(instrs, i, m2_name)
            m2_readers = _reg_readers(instrs, i + 1, m2_end, m2_name)
            if len(m2_readers) != 1:
                i += 1
                continue
            cidx = m2_readers[0]
            cadd = instrs[cidx]
            if cadd.op != "add" or not _is_i64(cadd) or cadd.mods:
                i += 1
                continue
            if (cadd.pred, cadd.neg) != pred:
                i += 1
                continue
            if len(cadd.srcs or []) != 2 or not isinstance(cadd.dest, RegOp):
                i += 1
                continue
            ca, cb = cadd.srcs
            if isinstance(ca, RegOp) and ca.name == m2_name and isinstance(cb, RegOp) \
                    and not isinstance(cb, VectorRegOp):
                b_reg = cb
            elif isinstance(cb, RegOp) and cb.name == m2_name and isinstance(ca, RegOp) \
                    and not isinstance(ca, VectorRegOp):
                b_reg = ca
            else:
                i += 1
                continue
            f_name = cadd.dest.name

            # %F's unique reader in its window must be one global ld/st base.
            f_end = _next_write_after(instrs, cidx, f_name)
            if _reg_readers(instrs, cidx + 1, f_end, f_name):
                i += 1
                continue                  # %F used as a value, not just an address
            f_bases = _membase_readers(instrs, cidx + 1, f_end, f_name)
            if len(f_bases) != 1:
                i += 1
                continue
            ld_idx = f_bases[0]
            ldst = instrs[ld_idx]
            if ldst.op not in ('ld', 'st', 'atom') or 'global' not in (ldst.types or ()):
                i += 1
                continue

            b_name = b_reg.name

            # Offset to fold = k << S2, into the consuming memory operand.
            off = (K << S2) & 0xFFFFFFFFFFFFFFFF
            soff = off - (1 << 64) if off >= (1 << 63) else off
            tgt = None
            for si, s in enumerate(ldst.srcs or []):
                if isinstance(s, MemOp) and isinstance(s.base, str):
                    bn = s.base if s.base.startswith('%') else f'%{s.base}'
                    if bn == f_name:
                        tgt = (si, s)
                        break
            if tgt is None:
                i += 1
                continue
            si, memop = tgt
            new_off = memop.offset + soff
            if not (_OFF_MIN <= new_off <= _OFF_MAX):
                i += 1
                continue

            # Match ptxas: the LEA index is the INTERMEDIATE %M1 = X<<S1 (kept,
            # non-zero-extended), the scale is the OUTER shift S2, and limbs
            # sharing (M1, S2, B) collapse to ONE shared base  B + M1<<S2  +
            # folded per-limb offsets.  Use a fresh base vreg so the shared
            # address is never clobbered by FORGE's vreg reuse.
            key = (m1_name, S2, b_name, pred)
            cached = cse.get(key)
            if cached is not None:
                fsh, anchor_i = cached
                # M1 and B must hold the SAME value at anchor and here.
                if (_writes_between(instrs, anchor_i, i, m1_name)
                        or _writes_between(instrs, anchor_i, i, b_name)):
                    i += 1
                    continue
                # Reuse the shared base; drop this limb's whole address chain.
                ldst.srcs[si] = MemOp(base=fsh, offset=new_off)
                dead.add(id(instrs[i]))            # outer shl
                dead.add(id(cadd))                 # consumer add
                if not is_k0:
                    dead.add(id(a_def))            # leading add
            else:
                # Anchor: emit  shl %Mn,%M1,S2 ; add %Fsh,%B,%Mn  (fresh vregs),
                # fold this limb's own offset into its memop.
                mn = _alloc_vreg(fn)
                fsh = _alloc_vreg(fn)
                instrs[i] = Instruction(op="shl", types=["b64"], dest=RegOp(mn),
                                        srcs=[RegOp(m1_name), ImmOp(S2)],
                                        pred=instrs[i].pred, neg=instrs[i].neg)
                cadd.dest = RegOp(fsh)
                cadd.srcs = [RegOp(b_name), RegOp(mn)]
                ldst.srcs[si] = MemOp(base=fsh, offset=new_off)
                if not is_k0:
                    dead.add(id(a_def))            # leading add now dead
                cse[key] = (fsh, i)
            n += 1
            i += 1
        if dead:
            bb.instructions = [x for x in instrs if id(x) not in dead]
    return n


def run(mod) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
