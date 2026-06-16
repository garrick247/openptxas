"""SASS-level single-thread CPU emulator for OpenPTXas cubins.

Purpose: run cubins on the CPU and dump per-instruction register / memory
traces, so codegen bugs can be diagnosed without burning time on cuda-gdb.

Decoder strategy: parse nvdisasm's text output (avoids reimplementing the
SASS bit-level decoder).  Each line becomes an Instruction(pc, mnemonic,
operands).

State: GPRs (R0..R254, R255=RZ), URs (UR0..UR62, UR63=URZ), predicates
(P0..P6, P7=PT, UP0..UP6, UP7=UPT), per-thread frame, simulated global
memory, constant bank c[0] (params + descriptors), special registers.

This is the MVP — 35 opcodes, single-thread, no scoreboard / control bytes.
Good enough to catch wild address bugs by stepping through instructions
and seeing where R36 / R38 / R8 ends up holding unexpected values.
"""
from __future__ import annotations
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


NVDISASM = "/usr/local/cuda-13.3/bin/nvdisasm"


# ============================================================================
# ELF cubin parser — extract .text.<kernel> and .nv.constant0.<kernel>
# ============================================================================

def parse_cubin(path: str) -> dict:
    """Parse cubin ELF and return a dict mapping section names to bytes."""
    data = open(path, "rb").read()
    # ELF64 header layout (little-endian)
    assert data[:4] == b"\x7fELF", "Not an ELF file"
    shoff = struct.unpack("<Q", data[0x28:0x30])[0]
    shentsize = struct.unpack("<H", data[0x3a:0x3c])[0]
    shnum = struct.unpack("<H", data[0x3c:0x3e])[0]
    shstrndx = struct.unpack("<H", data[0x3e:0x40])[0]
    shstr_off = struct.unpack("<Q", data[shoff + shstrndx*shentsize + 0x18:shoff + shstrndx*shentsize + 0x20])[0]

    sections: dict[str, bytes] = {}
    for i in range(shnum):
        s_off = shoff + i * shentsize
        name_off = struct.unpack("<I", data[s_off:s_off+4])[0]
        sh_offset = struct.unpack("<Q", data[s_off+0x18:s_off+0x20])[0]
        sh_size = struct.unpack("<Q", data[s_off+0x20:s_off+0x28])[0]
        end = data.find(b"\x00", shstr_off + name_off)
        name = data[shstr_off + name_off:end].decode("ascii", "replace")
        sections[name] = data[sh_offset:sh_offset + sh_size]
    return sections


def find_kernel_name(sections: dict) -> str:
    """Find the kernel name from section names like '.text.kernel_name'."""
    for sname in sections:
        if sname.startswith(".text.") and not sname.startswith(".text..") and len(sname) > 6:
            return sname[6:]
    raise ValueError("No kernel .text section found")


# ============================================================================
# Decoder — parse nvdisasm output into Instruction objects
# ============================================================================

@dataclass
class Operand:
    """One operand of a SASS instruction."""
    raw: str
    # Decoded fields (set when parsed):
    kind: str = "unknown"   # 'reg', 'ureg', 'pred', 'upred', 'imm', 'cbank', 'desc', 'mem', 'label', 'special'
    reg_idx: Optional[int] = None
    is_64: bool = False     # R.64 / UR.64 / R-pair
    neg: bool = False       # -R or ~R
    invert: bool = False    # ~R (bitwise)
    imm_val: Optional[int] = None
    cbank_bank: Optional[int] = None
    cbank_off: Optional[int] = None
    mem_base: Optional[str] = None  # for desc[UR][R.64+off]
    mem_off: int = 0
    label: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Instruction:
    pc: int
    pred: Optional[str]   # e.g. "@P0", "@!P0", "@!UP0", None
    mnemonic: str
    operands: list[Operand]
    raw_text: str         # original nvdisasm line for debugging


_REG_RE = re.compile(r"^(-|~)?(R|UR|P|UP)(\d+|Z|T)(\.64|\.X|\.HI|\.LO)?$")
_LABEL_RE = re.compile(r"^`\((\.L_\w+)\)$")
_CBANK_RE = re.compile(r"^c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]$")
_IMM_RE = re.compile(r"^(-?0x[0-9a-f]+|-?\d+)$", re.IGNORECASE)


def _parse_reg_token(tok: str) -> Optional[Operand]:
    """Try to parse a token as a register operand."""
    m = _REG_RE.match(tok)
    if not m:
        return None
    sign, kind_letter, idx_str, suffix = m.groups()
    op = Operand(raw=tok)
    op.neg = (sign == "-")
    op.invert = (sign == "~")
    if idx_str == "Z":
        op.reg_idx = 255 if kind_letter in ("R", "UR") else 7
    elif idx_str == "T":
        op.reg_idx = 7  # PT
    else:
        op.reg_idx = int(idx_str)
    if kind_letter == "R":
        op.kind = "reg"
    elif kind_letter == "UR":
        op.kind = "ureg"
    elif kind_letter == "P":
        op.kind = "pred"
    elif kind_letter == "UP":
        op.kind = "upred"
    if suffix == ".64":
        op.is_64 = True
    elif suffix == ".HI":
        op.extra["half"] = "hi"
    elif suffix == ".LO":
        op.extra["half"] = "lo"
    elif suffix == ".X":
        op.extra["carry_in"] = True
    return op


def _parse_operand(tok: str) -> Operand:
    """Parse a single operand token."""
    tok = tok.strip()
    # Special markers
    if tok in ("RZ", "URZ", "PT", "UPT", "!PT"):
        op = Operand(raw=tok, kind="special")
        if tok == "RZ":
            op.kind = "reg"
            op.reg_idx = 255
        elif tok == "URZ":
            op.kind = "ureg"
            op.reg_idx = 63
        elif tok == "PT":
            op.kind = "pred"
            op.reg_idx = 7
        elif tok == "UPT":
            op.kind = "upred"
            op.reg_idx = 7
        elif tok == "!PT":
            op.kind = "pred"
            op.reg_idx = 7
            op.neg = True
        return op
    if tok.startswith("!P") or tok.startswith("!UP"):
        sub = tok[1:]
        sub_op = _parse_reg_token(sub)
        if sub_op:
            sub_op.neg = True
            sub_op.raw = tok
            return sub_op
    # Special source registers (SR_TID.X etc.)
    if tok.startswith("SR_"):
        return Operand(raw=tok, kind="special", label=tok)
    # Register or register-with-modifier
    reg = _parse_reg_token(tok)
    if reg:
        return reg
    # Constant bank: c[0x0][0x37c]
    m = _CBANK_RE.match(tok)
    if m:
        return Operand(raw=tok, kind="cbank",
                       cbank_bank=int(m.group(1), 16),
                       cbank_off=int(m.group(2), 16))
    # Immediate
    if _IMM_RE.match(tok):
        return Operand(raw=tok, kind="imm",
                       imm_val=int(tok, 0))
    # Label reference: `(.L_x_0)
    m = _LABEL_RE.match(tok)
    if m:
        return Operand(raw=tok, kind="label", label=m.group(1))
    # Memory reference: [R20] or [R10.64+0x4] or desc[UR4][R22.64]
    if tok.startswith("[") and tok.endswith("]"):
        # Strip outer brackets; the format may be `[R.64+offset]`
        inner = tok[1:-1]
        # Try splitting on '+'
        op = Operand(raw=tok, kind="mem")
        if "+" in inner:
            a, b = inner.rsplit("+", 1)
            base_op = _parse_reg_token(a.strip())
            if base_op:
                op.mem_base = a.strip()
            else:
                op.mem_base = a.strip()
            try:
                op.mem_off = int(b.strip(), 0)
            except ValueError:
                op.mem_off = 0
        else:
            op.mem_base = inner.strip()
        return op
    # Descriptor-based memory: desc[URn][...]
    if tok.startswith("desc[") and "][" in tok:
        return Operand(raw=tok, kind="desc")
    # Anything else: store raw for later
    return Operand(raw=tok, kind="unknown")


def decode_kernel(cubin_path: str, kernel_name: str) -> list[Instruction]:
    """Run nvdisasm on the cubin and parse the kernel's instructions."""
    out = subprocess.check_output([NVDISASM, cubin_path], stderr=subprocess.DEVNULL).decode("utf-8", "replace")
    lines = out.splitlines()
    instrs: list[Instruction] = []
    in_kernel = False
    pending_label: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if line.endswith(":") and (line == f"{kernel_name}:" or line == f".text.{kernel_name}:"):
            in_kernel = True
            continue
        if in_kernel:
            # Stop at the next section marker or blank-line-after-EXIT chain.
            if line.startswith(".section") or line.startswith("//-"):
                break
            if not line:
                continue
            # Label line: ".L_x_2:"
            if line.startswith(".L_") and line.endswith(":"):
                pending_label = line[:-1]
                continue
            # Instruction line: "/*0010*/  IADD3 R...".
            m = re.match(r"^/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;?$", line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            rest = m.group(2)
            # Strip trailing semicolon and comments
            rest = rest.rstrip(";").strip()
            # Predicate prefix?
            pred = None
            pm = re.match(r"^(@!?(?:P|UP)\d)\s+(.*)$", rest)
            if pm:
                pred = pm.group(1)
                rest = pm.group(2)
            # Split mnemonic from operands
            parts = rest.split(None, 1)
            mnem = parts[0]
            operands: list[Operand] = []
            if len(parts) > 1:
                # Split operands on commas (but not inside brackets).
                opstr = parts[1]
                op_tokens = _split_operands(opstr)
                operands = [_parse_operand(t) for t in op_tokens]
            instrs.append(Instruction(pc=pc, pred=pred, mnemonic=mnem,
                                      operands=operands, raw_text=line))
            if pending_label:
                instrs[-1].operands.insert(0, Operand(raw=f"_label_={pending_label}",
                                                     kind="label_marker",
                                                     label=pending_label))
                pending_label = None
    return instrs


def _split_operands(opstr: str) -> list[str]:
    """Split operands on top-level commas (not inside brackets)."""
    toks = []
    depth = 0
    cur = []
    for ch in opstr:
        if ch in "[(":
            depth += 1
            cur.append(ch)
        elif ch in "])":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            toks.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        toks.append("".join(cur).strip())
    return [t for t in toks if t]


# ============================================================================
# State model — registers, memory, predicates, constants
# ============================================================================

@dataclass
class State:
    gprs: list[int]                      # 256 × 32-bit
    uregs: list[int]                     # 64 × 32-bit
    preds: list[bool]                    # 8 (P0..P7=PT)
    upreds: list[bool]                   # 8 (UP0..UP7=UPT)
    pc: int = 0                          # byte offset into .text
    halted: bool = False
    exited: bool = False
    # Memory:
    cbank: bytes = b""                   # c[0] contents (params + descriptors)
    local_frame: bytearray = field(default_factory=lambda: bytearray(8192))
    globals: dict[int, bytearray] = field(default_factory=dict)  # base addr -> bytes
    # Thread/block context:
    tid: tuple[int, int, int] = (0, 0, 0)
    ctaid: tuple[int, int, int] = (0, 0, 0)
    ntid: tuple[int, int, int] = (1, 1, 1)
    # Special-reg base addr for the per-thread local segment (R1's initial value)
    local_base: int = 0x100000   # arbitrary; just a base for STL/LDL addressing


def new_state(num_gprs: int = 256) -> State:
    s = State(
        gprs=[0] * 256,
        uregs=[0] * 64,
        preds=[False] * 8,
        upreds=[False] * 8,
    )
    # RZ, URZ are conceptually always 0; we just never write them.
    # PT, UPT are conceptually always True.
    s.preds[7] = True
    s.upreds[7] = True
    return s


def read_gpr(state: State, idx: int) -> int:
    if idx == 255:
        return 0
    return state.gprs[idx] & 0xFFFFFFFF


def write_gpr(state: State, idx: int, val: int) -> None:
    if idx == 255:
        return
    state.gprs[idx] = val & 0xFFFFFFFF


def read_ureg(state: State, idx: int) -> int:
    if idx == 63:
        return 0
    return state.uregs[idx] & 0xFFFFFFFF


def write_ureg(state: State, idx: int, val: int) -> None:
    if idx == 63:
        return
    state.uregs[idx] = val & 0xFFFFFFFF


def read_pred(state: State, idx: int, neg: bool = False) -> bool:
    if idx == 7:
        v = True
    else:
        v = state.preds[idx]
    return (not v) if neg else v


def read_upred(state: State, idx: int, neg: bool = False) -> bool:
    if idx == 7:
        v = True
    else:
        v = state.upreds[idx]
    return (not v) if neg else v


def read_op(state: State, op: Operand) -> int:
    """Read an operand's value (for arithmetic/logic instructions)."""
    if op.kind == "reg":
        v = read_gpr(state, op.reg_idx)
    elif op.kind == "ureg":
        v = read_ureg(state, op.reg_idx)
    elif op.kind == "imm":
        v = op.imm_val & 0xFFFFFFFF if op.imm_val >= 0 else op.imm_val & 0xFFFFFFFF
    elif op.kind == "cbank":
        # Read from cbank c[bank][off] (32-bit)
        off = op.cbank_off
        if off + 4 > len(state.cbank):
            return 0
        v = struct.unpack_from("<I", state.cbank, off)[0]
    else:
        v = 0
    if op.neg:
        v = (-v) & 0xFFFFFFFF
    if op.invert:
        v = (~v) & 0xFFFFFFFF
    return v


def read_op_pair(state: State, op: Operand) -> tuple[int, int]:
    """Read a 64-bit reg pair as (low, high)."""
    if op.kind == "reg":
        if op.reg_idx == 255:
            return 0, 0
        lo = read_gpr(state, op.reg_idx)
        hi = read_gpr(state, op.reg_idx + 1)
    elif op.kind == "ureg":
        if op.reg_idx == 63:
            return 0, 0
        lo = read_ureg(state, op.reg_idx)
        hi = read_ureg(state, op.reg_idx + 1)
    elif op.kind == "cbank":
        off = op.cbank_off
        if off + 8 > len(state.cbank):
            return 0, 0
        lo, hi = struct.unpack_from("<II", state.cbank, off)
    elif op.kind == "imm":
        v = op.imm_val & 0xFFFFFFFFFFFFFFFF
        lo = v & 0xFFFFFFFF
        hi = (v >> 32) & 0xFFFFFFFF
    else:
        lo = hi = 0
    return lo, hi
"""
Append to sass_emu.py: opcode semantics + execution loop + memory model.

The semantics are organized as a dispatch table indexed by mnemonic.
Each handler returns (next_pc, halt_reason) or None for normal flow.
"""

# ============================================================================
# Memory model — global / local / constant accesses
# ============================================================================

def read_mem_u32(state, addr: int) -> int:
    """Read a u32 from simulated global memory (returns 0 if unmapped)."""
    for base, buf in state.globals.items():
        if base <= addr < base + len(buf) and addr + 4 <= base + len(buf):
            off = addr - base
            return struct.unpack_from("<I", buf, off)[0]
    return 0


def write_mem_u32(state, addr: int, val: int) -> None:
    for base, buf in state.globals.items():
        if base <= addr < base + len(buf) and addr + 4 <= base + len(buf):
            off = addr - base
            struct.pack_into("<I", buf, off, val & 0xFFFFFFFF)
            return
    # Out-of-bounds: log it
    state.events.append(("OOB_WRITE", addr, val))


def addr_in_any_buffer(state, addr: int, size: int) -> Optional[tuple[int, int]]:
    """Return (base, offset) if address fits in a mapped buffer, else None."""
    for base, buf in state.globals.items():
        if base <= addr and addr + size <= base + len(buf):
            return base, addr - base
    return None


# ============================================================================
# Bitwise / SHF helpers
# ============================================================================

def shf_l_u64_hi(lo: int, hi: int, shift: int, src_hi_reg: int) -> int:
    """SHF.L.U64.HI dst, lo, shift, hi_src: dst = high half of ((hi_src<<32 | lo) << shift)."""
    # SASS: result_64 = ((hi_src << 32) | lo) << shift; take high 32 bits.
    combined = (src_hi_reg & 0xFFFFFFFF) << 32 | (lo & 0xFFFFFFFF)
    return ((combined << shift) >> 32) & 0xFFFFFFFF


def shf_r_u32_hi(lo: int, shift: int, src_hi_reg: int) -> int:
    """SHF.R.U32.HI dst, lo, shift, hi_src: high 32 bits of ((hi<<32|lo) >> shift)."""
    combined = (src_hi_reg & 0xFFFFFFFF) << 32 | (lo & 0xFFFFFFFF)
    return ((combined >> shift) >> 0) & 0xFFFFFFFF if shift >= 32 else (combined >> shift) >> 32 & 0xFFFFFFFF


def shf_r_u32_hi_simple(src: int, shift: int) -> int:
    """SHF.R.U32.HI dst, RZ, shift, src: dst = src >> shift."""
    return (src >> shift) & 0xFFFFFFFF


# ============================================================================
# LOP3 lookup-table evaluator
# ============================================================================

def lop3(a: int, b: int, c: int, lut: int) -> int:
    """LOP3.LUT(a, b, c, lut): evaluate the 256-entry truth table.
    For each output bit i: out[i] = lut bit at position (a[i]<<2 | b[i]<<1 | c[i])."""
    result = 0
    for i in range(32):
        ab = (a >> i) & 1
        bb = (b >> i) & 1
        cb = (c >> i) & 1
        sel = (ab << 2) | (bb << 1) | cb
        result |= ((lut >> sel) & 1) << i
    return result & 0xFFFFFFFF


# ============================================================================
# Opcode handlers — return (next_pc, halt_reason)
# ============================================================================

def _op_LDC(state, ins, ctx):
    """LDC dest, c[bank][off] — 32-bit constant load."""
    dest, cb = ins.operands
    val = read_op(state, cb)
    write_gpr(state, dest.reg_idx, val)


def _op_LDC_64(state, ins, ctx):
    """LDC.64 dest, c[bank][off] — 64-bit constant load to reg pair."""
    dest, cb = ins.operands
    lo, hi = read_op_pair(state, cb)
    write_gpr(state, dest.reg_idx, lo)
    write_gpr(state, dest.reg_idx + 1, hi)


def _op_LDCU(state, ins, ctx):
    """LDCU dest_ureg, c[bank][off] — 32-bit uniform constant load."""
    dest, cb = ins.operands
    val = read_op(state, cb)
    write_ureg(state, dest.reg_idx, val)


def _op_LDCU_64(state, ins, ctx):
    """LDCU.64 dest_ureg, c[bank][off]."""
    dest, cb = ins.operands
    lo, hi = read_op_pair(state, cb)
    write_ureg(state, dest.reg_idx, lo)
    write_ureg(state, dest.reg_idx + 1, hi)


def _op_LDCU_128(state, ins, ctx):
    """LDCU.128 dest_ureg, c[bank][off]."""
    dest, cb = ins.operands
    off = cb.cbank_off
    for i in range(4):
        if off + i*4 + 4 > len(state.cbank):
            val = 0
        else:
            val = struct.unpack_from("<I", state.cbank, off + i*4)[0]
        write_ureg(state, dest.reg_idx + i, val)


def _op_S2R(state, ins, ctx):
    """S2R dest, SR_TID.X / SR_TID.Y / SR_TID.Z."""
    dest, src = ins.operands
    src_name = src.label or src.raw
    val = 0
    if src_name == "SR_TID.X":
        val = state.tid[0]
    elif src_name == "SR_TID.Y":
        val = state.tid[1]
    elif src_name == "SR_TID.Z":
        val = state.tid[2]
    elif src_name == "SR_CTAID.X":
        val = state.ctaid[0]
    elif src_name == "SR_CTAID.Y":
        val = state.ctaid[1]
    elif src_name == "SR_CTAID.Z":
        val = state.ctaid[2]
    elif src_name == "SR_NTID.X":
        val = state.ntid[0]
    write_gpr(state, dest.reg_idx, val)


def _op_S2UR(state, ins, ctx):
    """S2UR dest_ureg, SR_*."""
    dest, src = ins.operands
    src_name = src.label or src.raw
    val = 0
    if src_name == "SR_TID.X":
        val = state.tid[0]
    elif src_name == "SR_CTAID.X":
        val = state.ctaid[0]
    elif src_name == "SR_CTAID.Y":
        val = state.ctaid[1]
    elif src_name == "SR_NTID.X":
        val = state.ntid[0]
    write_ureg(state, dest.reg_idx, val)


def _op_MOV(state, ins, ctx):
    """MOV dest, src — 32-bit move."""
    dest, src = ins.operands
    write_gpr(state, dest.reg_idx, read_op(state, src))


def _op_NOP(state, ins, ctx):
    pass


def _op_IADD3(state, ins, ctx):
    """IADD3 dest [, P_carry_out, P_borrow], src0, src1, src2 — 3-input add."""
    dest = ins.operands[0]
    # Skip optional predicate dest operands (P0, PT).
    src_ops = [o for o in ins.operands[1:] if o.kind not in ("pred",)]
    # Pad to 3 sources.
    while len(src_ops) < 3:
        src_ops.append(Operand(raw="RZ", kind="reg", reg_idx=255))
    src_vals = [read_op(state, s) for s in src_ops]
    total = (src_vals[0] + src_vals[1] + src_vals[2]) & 0xFFFFFFFFFFFFFFFF
    write_gpr(state, dest.reg_idx, total & 0xFFFFFFFF)
    # Carry out: bit 32 of the 64-bit sum.
    carry = (total >> 32) & 1
    # Find which pred operands are dests; set them to the carry.
    for op in ins.operands[1:]:
        if op.kind == "pred" and not op.neg and op.reg_idx != 7:
            state.preds[op.reg_idx] = bool(carry)
            break


def _op_IADD3_X(state, ins, ctx):
    """IADD3.X dest, src0, src1, src2, P_carry_in[, !PT] — 3-input add with carry-in."""
    dest = ins.operands[0]
    # Operands after dest: srcs, then 1-2 predicate inputs.
    src_ops = []
    pred_in = None
    pred_neg = False
    for op in ins.operands[1:]:
        if op.kind == "pred":
            if pred_in is None and op.reg_idx != 7:
                pred_in = op.reg_idx
                pred_neg = op.neg
            # Skip remaining preds (typically !PT).
        else:
            src_ops.append(op)
    while len(src_ops) < 3:
        src_ops.append(Operand(raw="RZ", kind="reg", reg_idx=255))
    src_vals = [read_op(state, s) for s in src_ops]
    carry_in = 0
    if pred_in is not None:
        carry_in = 1 if read_pred(state, pred_in, pred_neg) else 0
    total = (src_vals[0] + src_vals[1] + src_vals[2] + carry_in) & 0xFFFFFFFFFFFFFFFF
    write_gpr(state, dest.reg_idx, total & 0xFFFFFFFF)


def _op_IADD(state, ins, ctx):
    """IADD dest, src0, src1 — simple 2-input add."""
    dest = ins.operands[0]
    src_ops = ins.operands[1:]
    vals = [read_op(state, s) for s in src_ops]
    total = sum(vals) & 0xFFFFFFFF
    write_gpr(state, dest.reg_idx, total)


def _op_IADD_64(state, ins, ctx):
    """IADD.64 dest_pair, src0_pair, src1_pair."""
    dest = ins.operands[0]
    src_ops = ins.operands[1:]
    lo, hi = 0, 0
    for s in src_ops:
        slo, shi = read_op_pair(state, s)
        if s.neg:
            v = ((shi << 32) | slo)
            v = (-v) & 0xFFFFFFFFFFFFFFFF
            slo = v & 0xFFFFFFFF
            shi = (v >> 32) & 0xFFFFFFFF
        total64 = (lo + (hi << 32) + slo + (shi << 32))
        lo = total64 & 0xFFFFFFFF
        hi = (total64 >> 32) & 0xFFFFFFFF
    write_gpr(state, dest.reg_idx, lo)
    write_gpr(state, dest.reg_idx + 1, hi)


def _op_IMAD(state, ins, ctx):
    """IMAD dest, a, b, c — dest = a*b + c (32-bit)."""
    dest = ins.operands[0]
    a, b, c = ins.operands[1:4]
    av = read_op(state, a)
    bv = read_op(state, b)
    cv = read_op(state, c)
    res = (av * bv + cv) & 0xFFFFFFFF
    write_gpr(state, dest.reg_idx, res)


def _op_IMAD_SHL_U32(state, ins, ctx):
    """IMAD.SHL.U32 dest, src, K, addend - WIDE write of src*K + addend.

    Per encoder docstring: imm16 = (1<<shift_amount); the disasm shows
    that imm16 directly. So semantically: dest_pair = src32 * K + addend64.
    Writes BOTH dest and dest+1.
    """
    dest = ins.operands[0]
    src, k_op, addend = ins.operands[1:4]
    sv = read_op(state, src)
    kv = read_op(state, k_op)
    a_lo, a_hi = read_op_pair(state, addend) if addend.kind != "imm" else (read_op(state, addend), 0)
    if addend.kind == "reg" and addend.reg_idx == 255:
        a_lo = a_hi = 0
    res64 = (sv * kv + ((a_hi << 32) | a_lo)) & 0xFFFFFFFFFFFFFFFF
    write_gpr(state, dest.reg_idx, res64 & 0xFFFFFFFF)
    write_gpr(state, dest.reg_idx + 1, (res64 >> 32) & 0xFFFFFFFF)


def _op_IMAD_WIDE(state, ins, ctx):
    """IMAD.WIDE dest_pair, a, b, c_pair — dest64 = a*b + c64."""
    dest = ins.operands[0]
    a, b = ins.operands[1:3]
    c_op = ins.operands[3] if len(ins.operands) > 3 else None
    av = read_op(state, a)
    bv = read_op(state, b)
    res64 = av * bv
    if c_op is not None:
        clo, chi = read_op_pair(state, c_op)
        res64 += (chi << 32) | clo
    res64 &= 0xFFFFFFFFFFFFFFFF
    write_gpr(state, dest.reg_idx, res64 & 0xFFFFFFFF)
    write_gpr(state, dest.reg_idx + 1, (res64 >> 32) & 0xFFFFFFFF)


def _op_IMAD_WIDE_U32(state, ins, ctx):
    return _op_IMAD_WIDE(state, ins, ctx)


def _op_ISETP_GE_U32_AND(state, ins, ctx):
    """ISETP.GE.U32.AND P_out, PT, a, b, PT — set P_out = (a >= b) AND PT."""
    p_out = ins.operands[0]
    # Skip the PT slot, then a, b.
    # Common form: P0, PT, R, R_or_UR, PT
    operands = ins.operands
    # Find the first numeric operand after PT.
    a = b = None
    seen_pt = False
    for op in operands[1:]:
        if op.kind == "pred" and op.reg_idx == 7 and not op.neg:
            seen_pt = True
            continue
        if a is None:
            a = op
        elif b is None:
            b = op
            break
    av = read_op(state, a)
    bv = read_op(state, b)
    state.preds[p_out.reg_idx] = (av >= bv)


def _op_ISETP_GE_U64_AND(state, ins, ctx):
    p_out = ins.operands[0]
    # Find first two non-PT operands.
    others = [o for o in ins.operands[1:] if not (o.kind == "pred" and o.reg_idx == 7)]
    a, b = others[0], others[1]
    a_lo, a_hi = read_op_pair(state, a)
    b_lo, b_hi = read_op_pair(state, b)
    av = (a_hi << 32) | a_lo
    bv = (b_hi << 32) | b_lo
    state.preds[p_out.reg_idx] = (av >= bv)


def _op_ISETP_LT_U32_AND(state, ins, ctx):
    p_out = ins.operands[0]
    others = [o for o in ins.operands[1:] if not (o.kind == "pred" and o.reg_idx == 7)]
    a, b = others[0], others[1]
    state.preds[p_out.reg_idx] = (read_op(state, a) < read_op(state, b))


def _op_ISETP_GT_U32_AND(state, ins, ctx):
    p_out = ins.operands[0]
    others = [o for o in ins.operands[1:] if not (o.kind == "pred" and o.reg_idx == 7)]
    a, b = others[0], others[1]
    state.preds[p_out.reg_idx] = (read_op(state, a) > read_op(state, b))


def _op_ISETP_EQ_U32_AND(state, ins, ctx):
    p_out = ins.operands[0]
    others = [o for o in ins.operands[1:] if not (o.kind == "pred" and o.reg_idx == 7)]
    a, b = others[0], others[1]
    state.preds[p_out.reg_idx] = (read_op(state, a) == read_op(state, b))


def _op_SHF_L_U64_HI(state, ins, ctx):
    """SHF.L.U64.HI dest, lo, shift, hi — high half of (hi:lo << shift)."""
    dest, lo_op, sh_op, hi_op = ins.operands
    res = shf_l_u64_hi(read_op(state, lo_op), 0, read_op(state, sh_op), read_op(state, hi_op))
    write_gpr(state, dest.reg_idx, res)


def _op_SHF_R_U32_HI(state, ins, ctx):
    """SHF.R.U32.HI dest, RZ, shift, src — dest = src >> shift."""
    dest = ins.operands[0]
    operands = ins.operands[1:]
    # Layout: RZ, shift_imm, src (we read shift as op[1], src as op[2]).
    sh = read_op(state, operands[1])
    src = read_op(state, operands[2])
    write_gpr(state, dest.reg_idx, (src >> sh) & 0xFFFFFFFF)


def _op_SHF_R_U64(state, ins, ctx):
    """SHF.R.U64 dest, lo, shift, hi — low half of (hi:lo >> shift)."""
    dest, lo_op, sh_op, hi_op = ins.operands
    lo = read_op(state, lo_op)
    sh = read_op(state, sh_op)
    hi = read_op(state, hi_op)
    combined = (hi << 32) | lo
    write_gpr(state, dest.reg_idx, (combined >> sh) & 0xFFFFFFFF)


def _op_LOP3_LUT(state, ins, ctx):
    """LOP3.LUT dest, a, b, c, lut, !PT — 3-input bitwise via 8-bit lookup table."""
    dest = ins.operands[0]
    a, b, c, lut = ins.operands[1:5]
    av = read_op(state, a)
    bv = read_op(state, b)
    cv = read_op(state, c)
    lv = read_op(state, lut) & 0xFF
    write_gpr(state, dest.reg_idx, lop3(av, bv, cv, lv))


def _op_LDG_E(state, ins, ctx):
    """LDG.E dest, desc[UR][R.64+off] — global 32-bit load."""
    dest = ins.operands[0]
    desc_op = ins.operands[1]
    # Parse desc[URd][R.64+off]
    m = re.match(r"^desc\[UR(\d+)\]\[(R\d+|RZ)\.64(?:\+0x([0-9a-f]+))?\]$", desc_op.raw)
    if not m:
        m = re.match(r"^desc\[UR(\d+)\]\[(R\d+|RZ)\.64(?:\+(\d+))?\]$", desc_op.raw)
    if not m:
        state.events.append(("LDG_DECODE_FAIL", desc_op.raw))
        return
    ur_idx = int(m.group(1))
    reg_str = m.group(2)
    off_str = m.group(3) or "0"
    off = int(off_str, 16) if off_str.startswith("0x") or (m.re.pattern.find("0x") > 0) else int(off_str, 0) if off_str else 0
    try:
        off = int(off_str, 0) if off_str else 0
    except ValueError:
        off = 0
    if reg_str == "RZ":
        addr_lo = 0
        addr_hi = 0
    else:
        reg_idx = int(reg_str[1:])
        addr_lo = read_gpr(state, reg_idx)
        addr_hi = read_gpr(state, reg_idx + 1)
    addr = (addr_hi << 32) | addr_lo
    addr += off
    # Check alignment
    if addr & 0x3:
        state.events.append(("LDG_MISALIGNED", ins.pc, addr))
    res = read_mem_u32(state, addr)
    if addr_in_any_buffer(state, addr, 4) is None:
        state.events.append(("LDG_OOB", ins.pc, addr))
    write_gpr(state, dest.reg_idx, res)


def _op_STG_E(state, ins, ctx):
    desc_op = ins.operands[0]
    src = ins.operands[1]
    m = re.match(r"^desc\[UR(\d+)\]\[(R\d+|RZ)\.64(?:\+0x([0-9a-f]+))?\]$", desc_op.raw)
    if not m:
        return
    reg_str = m.group(2)
    off_str = m.group(3) or "0"
    try:
        off = int(off_str, 16)
    except ValueError:
        off = 0
    if reg_str == "RZ":
        addr_lo = addr_hi = 0
    else:
        reg_idx = int(reg_str[1:])
        addr_lo = read_gpr(state, reg_idx)
        addr_hi = read_gpr(state, reg_idx + 1)
    addr = (addr_hi << 32) | addr_lo
    addr += off
    val = read_op(state, src)
    if addr & 0x3:
        state.events.append(("STG_MISALIGNED", ins.pc, addr))
    if addr_in_any_buffer(state, addr, 4) is None:
        state.events.append(("STG_OOB", ins.pc, addr))
    write_mem_u32(state, addr, val)


def _op_LDL(state, ins, ctx):
    """LDL dest, [R_addr] — local load."""
    dest = ins.operands[0]
    mem = ins.operands[1]
    base = mem.mem_base
    off = mem.mem_off
    addr_reg = base.strip()
    if addr_reg.startswith("R"):
        ridx = 255 if addr_reg == "RZ" else int(addr_reg[1:])
        addr = read_gpr(state, ridx)
    else:
        addr = 0
    addr += off
    # Local frame is bytes 0..len(local_frame); use R1's initial as offset.
    # We use absolute-address semantic: subtract state.local_base.
    local_off = addr - state.local_base
    if local_off < 0 or local_off + 4 > len(state.local_frame):
        state.events.append(("LDL_OOB", ins.pc, addr, local_off))
        val = 0
    else:
        val = struct.unpack_from("<I", state.local_frame, local_off)[0]
    write_gpr(state, dest.reg_idx, val)


def _op_STL(state, ins, ctx):
    """STL [R_addr], R_src — local store."""
    mem = ins.operands[0]
    src = ins.operands[1]
    base = mem.mem_base.strip()
    off = mem.mem_off
    if base.startswith("R"):
        ridx = 255 if base == "RZ" else int(base[1:])
        addr = read_gpr(state, ridx)
    else:
        addr = 0
    addr += off
    val = read_op(state, src)
    local_off = addr - state.local_base
    if local_off < 0 or local_off + 4 > len(state.local_frame):
        state.events.append(("STL_OOB", ins.pc, addr, local_off))
        return
    struct.pack_into("<I", state.local_frame, local_off, val & 0xFFFFFFFF)


def _op_SEL_64(state, ins, ctx):
    """SEL.64 dest, a_pair, b_pair, pred — dest64 = pred ? b64 : a64 (or vice versa)."""
    dest = ins.operands[0]
    a, b = ins.operands[1], ins.operands[2]
    pred_op = ins.operands[3]
    p = read_pred(state, pred_op.reg_idx, pred_op.neg)
    a_lo, a_hi = read_op_pair(state, a)
    b_lo, b_hi = read_op_pair(state, b)
    # SASS convention: SEL.64 returns first operand if pred TRUE, else second.
    if p:
        write_gpr(state, dest.reg_idx, a_lo)
        write_gpr(state, dest.reg_idx + 1, a_hi)
    else:
        write_gpr(state, dest.reg_idx, b_lo)
        write_gpr(state, dest.reg_idx + 1, b_hi)


def _op_HFMA2(state, ins, ctx):
    # HFMA2 R_pair, ... — simplified: produce 0
    dest = ins.operands[0]
    write_gpr(state, dest.reg_idx, 0)
    write_gpr(state, dest.reg_idx + 1, 0)


def _op_BRA(state, ins, ctx):
    # If predicated and predicate is FALSE, fall through.
    if ins.pred:
        p_str = ins.pred
        neg = "!" in p_str
        m = re.match(r"^@(!?)(U?P)(\d+)$", p_str)
        if m:
            invert = (m.group(1) == "!")
            kind = m.group(2)
            idx = int(m.group(3))
            if kind == "P":
                cond = state.preds[idx] if idx != 7 else True
            else:
                cond = state.upreds[idx] if idx != 7 else True
            if invert:
                cond = not cond
            if not cond:
                return  # fall through
    # Take the branch.
    tgt_op = ins.operands[-1]
    if tgt_op.kind == "label":
        # Find PC of label in ctx.label_to_pc
        target_pc = ctx["label_to_pc"].get(tgt_op.label, None)
        if target_pc is not None:
            state.pc = target_pc - 16  # because execute() will add 16


def _op_BRA_U(state, ins, ctx):
    return _op_BRA(state, ins, ctx)


def _op_EXIT(state, ins, ctx):
    # If predicated and FALSE, don't exit.
    if ins.pred:
        m = re.match(r"^@(!?)(U?P)(\d+)$", ins.pred)
        if m:
            invert = (m.group(1) == "!")
            kind = m.group(2)
            idx = int(m.group(3))
            cond = state.preds[idx] if kind == "P" and idx != 7 else (
                state.upreds[idx] if kind == "UP" and idx != 7 else True)
            if invert:
                cond = not cond
            if not cond:
                return
    state.exited = True


OPCODE_DISPATCH = {
    "LDC": _op_LDC,
    "LDC.64": _op_LDC_64,
    "LDCU": _op_LDCU,
    "LDCU.64": _op_LDCU_64,
    "LDCU.128": _op_LDCU_128,
    "S2R": _op_S2R,
    "S2UR": _op_S2UR,
    "MOV": _op_MOV,
    "NOP": _op_NOP,
    "IADD3": _op_IADD3,
    "IADD3.X": _op_IADD3_X,
    "IADD": _op_IADD,
    "IADD.64": _op_IADD_64,
    "IMAD": _op_IMAD,
    "IMAD.SHL.U32": _op_IMAD_SHL_U32,
    "IMAD.WIDE": _op_IMAD_WIDE,
    "IMAD.WIDE.U32": _op_IMAD_WIDE_U32,
    "ISETP.GE.U32.AND": _op_ISETP_GE_U32_AND,
    "ISETP.GE.U64.AND": _op_ISETP_GE_U64_AND,
    "ISETP.LT.U32.AND": _op_ISETP_LT_U32_AND,
    "ISETP.GT.U32.AND": _op_ISETP_GT_U32_AND,
    "ISETP.EQ.U32.AND": _op_ISETP_EQ_U32_AND,
    "SHF.L.U64.HI": _op_SHF_L_U64_HI,
    "SHF.L.U32.HI": _op_SHF_L_U64_HI,  # same funnel (merkle hash rotates, shift<32)
    "SHF.R.U32.HI": _op_SHF_R_U32_HI,
    "SHF.R.U64": _op_SHF_R_U64,
    "LOP3.LUT": _op_LOP3_LUT,
    "LDG.E": _op_LDG_E,
    "STG.E": _op_STG_E,
    "LDL": _op_LDL,
    "STL": _op_STL,
    "SEL.64": _op_SEL_64,
    "HFMA2": _op_HFMA2,
    "BRA": _op_BRA,
    "BRA.U": _op_BRA_U,
    "EXIT": _op_EXIT,
}


# ============================================================================
# Execution loop
# ============================================================================

def run(state, instrs, max_steps: int = 100000, trace: bool = False) -> dict:
    """Execute instructions until EXIT or max_steps reached."""
    pc_to_idx = {ins.pc: i for i, ins in enumerate(instrs)}
    label_to_pc: dict[str, int] = {}
    for ins in instrs:
        for op in ins.operands:
            if op.kind == "label_marker" and op.label:
                label_to_pc[op.label] = ins.pc
    ctx = {"label_to_pc": label_to_pc, "pc_to_idx": pc_to_idx}
    state.events = []
    log: list[dict] = []

    steps = 0
    state.pc = 0
    while not state.exited and steps < max_steps:
        if state.pc not in pc_to_idx:
            state.events.append(("PC_BAD", state.pc))
            break
        ins = instrs[pc_to_idx[state.pc]]
        # Check predicate prefix: if predicate is false and op is NOT BRA/EXIT, skip.
        skip = False
        if ins.pred and ins.mnemonic not in ("BRA", "BRA.U", "EXIT"):
            m = re.match(r"^@(!?)(U?P)(\d+)$", ins.pred)
            if m:
                invert = (m.group(1) == "!")
                kind = m.group(2)
                idx = int(m.group(3))
                cond = (state.preds[idx] if kind == "P" and idx != 7
                        else (state.upreds[idx] if kind == "UP" and idx != 7 else True))
                if invert: cond = not cond
                if not cond:
                    skip = True
        if not skip:
            # Strip label_marker operands (they were attached for label tracking).
            real_ops = [o for o in ins.operands if o.kind != "label_marker"]
            if len(real_ops) != len(ins.operands):
                ins.operands = real_ops
            handler = OPCODE_DISPATCH.get(ins.mnemonic)
            if handler is None:
                state.events.append(("UNIMPLEMENTED", ins.pc, ins.mnemonic))
                # Treat as NOP and continue
            else:
                handler(state, ins, ctx)
        if trace:
            log.append({
                "pc": state.pc,
                "mnem": ins.mnemonic,
                "raw": ins.raw_text,
                "snapshot": _snapshot_relevant(state, ins),
            })
        state.pc += 16
        steps += 1
    return {"steps": steps, "events": state.events, "log": log,
            "exited": state.exited, "halted_pc": state.pc}


def _snapshot_relevant(state, ins) -> dict:
    """Record values of registers referenced by this instruction."""
    snap = {}
    for op in ins.operands:
        if op.kind == "reg" and op.reg_idx is not None:
            snap[f"R{op.reg_idx}"] = read_gpr(state, op.reg_idx)
            if op.is_64:
                snap[f"R{op.reg_idx+1}"] = read_gpr(state, op.reg_idx + 1)
        elif op.kind == "ureg" and op.reg_idx is not None:
            snap[f"UR{op.reg_idx}"] = read_ureg(state, op.reg_idx)
            if op.is_64:
                snap[f"UR{op.reg_idx+1}"] = read_ureg(state, op.reg_idx + 1)
        elif op.kind == "pred":
            snap[f"P{op.reg_idx}"] = state.preds[op.reg_idx] if op.reg_idx != 7 else True
    return snap
