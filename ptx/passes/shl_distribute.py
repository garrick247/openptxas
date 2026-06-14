"""
Offset-through-shift distribution for FORGE per-limb address chains.

ptxas 13.3 lowers the FORGE per-limb effective address
    base + (idx*2^S1 + k) * 2^S2
to a single shared base  base + idx*2^(S1+S2)  plus a constant memory
offset  k*2^S2  folded into each LDG/STG.  openptxas computes a full
address per limb instead, emitting an extra LEA pair (or IMAD.WIDE) per
limb where ptxas reuses one base + a cheap offset.

This pass performs the algebraically-exact PTX-IR rewrite.  Anchored on
the OUTER shift  shl %M2, %A, S2  (which unifies both forms):

  k != 0 (per-limb):
    shl %M1,%X,S1 ; add %A,%M1,k ; shl %M2,%A,S2 ; add %F,%B,%M2 ; ld/st [%F]
  k == 0 (slot 0):
    shl %M1,%X,S1 ;               shl %M2,%M1,S2 ; add %F,%B,%M2 ; ld/st [%F]

  =>  shl %Mn,%X,(S1+S2) ; add %F,%B,%Mn ; ld/st [%F + (k<<S2)]

using  (idx*2^S1 + k)*2^S2 == idx*2^(S1+S2) + k*2^S2  (exact mod 2^64).
The combined  %X<<(S1+S2)  then feeds the committed idx_hi_zero UR-base
LEA selection (X is the hi-zero loop index); k<<S2 folds into the memop.

# Block-local liveness (handles FORGE's heavily-reused vreg names)
Gating is computed per basic block over instruction windows, NOT via
function-wide def/use counts (which reject reused names outright):
- %M2's only reader in [def, next-redef) is the consumer add  -> safe to
  rebase the consumer and drop %M2's def.
- %F's only reader in its window is one global ld/st MemOp base  -> safe to
  fold the constant offset into exactly that memory op.
- k != 0: %A's only reader in its window is the outer shift  -> the leading
  add is dead after rewrite.  (%M1 may be shared across limbs; preserved.)
- k == 0: %M1 (= %A) may be shared; we never delete it, only add %Mn.
- %X must not be rewritten between its source shift and the insertion point.
- k<<S2 must fit the signed 24-bit LDG.E/STG.E offset; S1+S2 in [1,31].

Toggle: OFF unless OPENPTXAS_ENABLE_SHL_DISTRIBUTE is set (staged; gated so
the default path is byte-identical until GPU-validated on linux via
/tmp/driver.py — see project_openptxas.md).

FINDING 2026-06-13 — this is NOT the shape ptxas emits (do not chase
byte-parity with it).  ptxas 13.3 does NOT combine the two shifts: for
bit_reverse it keeps `R6 = revidx<<2` (IMAD.SHL + SHF.R.HI, so R6:R7 is
non-zero-extended) and emits `LEA R4, R6, base, 0x2` + `LEA.HI.X R5, R6,
base_hi, R7, 0x2` — i.e. a NON-HI-ZERO UR-base LEA on the *intermediate*
shifted index (src_c = R7 = the index's high half).  This pass instead
folds to `revidx<<(S1+S2)` (hi-zero), a valid but DIFFERENT cubin that
won't byte-match.  It also barely fires on real FORGE kernels because the
index vreg is typically clobbered before the address compute (the
stability check correctly bails).  The real path to ptxas-13.3 parity is
the non-hi-zero UR-base LEA selection (encode_lea_hi_x already supports
src_c=idx.hi via ur_base=True) + constant-offset folding that replicates
ptxas's per-site choices — NOT shift distribution.  Kept as a correct,
gated, isolation-tested transform (useful for a future non-byte-exact
instruction-reduction mode), not for parity.
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

            S = S1 + S2
            if S < 1 or S > 31:           # (1<<S) must stay u32 for the LEA scale chain
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

            # %X must be stable between its source shift and the insertion point i.
            x_src_idx = m1_def_idx if not is_k0 else a_def_idx
            if _last_write_before(instrs, i + 1, x_op.name) != \
                    _last_write_before(instrs, x_src_idx + 1, x_op.name):
                i += 1
                continue

            # Offset = K << S2, folded into the memory operand.
            off = (K << S2) & 0xFFFFFFFFFFFFFFFF
            soff = off - (1 << 64) if off >= (1 << 63) else off
            # Locate the target memop and its existing offset.
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

            # --- Rewrite -----------------------------------------------------
            mn = _alloc_vreg(fn)
            new_shl = Instruction(op="shl", types=["b64"], dest=RegOp(mn),
                                  srcs=[RegOp(x_op.name), ImmOp(S)],
                                  pred=instrs[i].pred, neg=instrs[i].neg)
            instrs[i] = new_shl                       # replace outer shift in place
            cadd.srcs = [RegOp(b_reg.name), RegOp(mn)]  # %F = %B + %Mn
            ldst.srcs[si] = MemOp(base=memop.base, offset=new_off)
            n += 1
            i += 1
    return n


def run(mod) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
