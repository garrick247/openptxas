"""Per-instruction GPU divergence locator for ours's merkle_hash_nodes.

Approach: PREFIX-TRUNCATION CHECKPOINT SWEEP.
  For a truncation byte-offset T, keep ours's SASS prefix [0:T] BYTE-IDENTICAL
  (so the scheduling of the instructions under test is not perturbed), then
  append a synthesized epilogue that dumps the GPR file R0..R(num_gprs-1) of
  THIS thread to its own slot in the parents buffer and EXITs.  Run on the GPU
  K times.  The emulator (deterministic, ideal-timing) gives the ground-truth
  register file at T.  Classify each checkpoint:
    DET-MATCH  : all K GPU runs agree AND equal the emulator   -> clean
    DET-DIVERGE: all K agree but != emulator                   -> LOGIC bug
    NONDET     : K runs disagree                               -> a race is
                 already manifest in some register by offset T
  The earliest NONDET checkpoint localizes where the hardware race first
  corrupts a register; diffing its register file against the previous clean
  checkpoint names the exact instruction(s) whose result went non-det.

Safety: the epilogue is built ENTIRELY from real templates lifted out of
ours's own SASS (IMAD.WIDE, STG.E, EXIT, NOP) with only operand bytes and a
conservative ctrl word (_patch_ctrl: stall=15, wbar=1, rbar=0x0f) patched in,
so encodings + ctrl bits are hardware-valid by construction.  Reuses R2 (gid)
and R6:R7 (parents base) — both proven live-to-end (the original final store
reads them).  Each thread dumps to parents[gid*STRIDE], so the 256 threads do
not race on the dump; we read gid=0's slot at offset 0.

Branch-safety: at N=256 every thread is valid (gid < n_parents) and takes the
straight-line .L_x_0 -> .L_x_1 path, so the only forward branch (BRA.U @0x110)
is dead.  Truncation anywhere in [0x120, first-STG) is therefore branch-safe.

Usage:
  merkle_locate.py validate          # sanity: dump at an early T, expect DET-MATCH
  merkle_locate.py scan [STRIDE]     # coarse sweep, find first NONDET window
  merkle_locate.py at <T_hex> [K]    # single checkpoint at byte offset T
"""
import sys, os, ctypes, struct, subprocess
sys.path[:0] = ["/home/garrick/forge-workbench", "/home/garrick/opencuda",
                "/home/garrick/openptxas"]
import workbench
import tools.sass_emu as E
import sass.pipeline as P
import cubin.emitter as EM
from sass.scoreboard import _patch_ctrl, _get_dest_regs, _get_src_regs
from dataclasses import replace

PTXPATH = "/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx"
PTX = open(PTXPATH).read()
KERNEL = "merkle_hash_nodes"
N = 256
SZ = 1 << 20
CHILD_BASE = 0x10000000
PAR_BASE = 0x20000000
children = bytes((i * 7 + 3) & 0xff for i in range(SZ))

# ---- conservative ctrl words (23-bit: stall[22:17] wbar[15] rbar[14:10] wdep[9:4] misc[3:0]) ----
CTRL_WAIT   = (0xf << 17) | (1 << 15) | (0x0f << 10) | (0x3f << 4) | 0x0  # NOP/STG/EXIT
CTRL_ALUW   = (0xf << 17) | (1 << 15) | (0x0f << 10) | (0x3e << 4) | 0x1  # IMAD.WIDE (writes GPR)

# ============================ capture ours's desc ============================
_CAP = {}
_ORIG_EMIT = EM.emit_cubin
def _cap_emit(desc):
    if desc.name == KERNEL:
        _CAP['desc'] = desc
    return _ORIG_EMIT(desc)
EM.emit_cubin = _cap_emit
P.emit_cubin = _cap_emit
P.compile_ptx_source(PTX)
EM.emit_cubin = _ORIG_EMIT
P.emit_cubin = _ORIG_EMIT
DESC = _CAP['desc']
SASS = DESC.sass_bytes
NG = DESC.num_gprs
print(f"[capture] ours {KERNEL}: {len(SASS)//16} instrs, num_gprs={NG}", file=sys.stderr)

# ============================ locate templates ============================
def _op(raw):  # 12-bit opcode = byte0 | (low nibble byte1 << 8)
    return raw[0] | ((raw[1] & 0x0f) << 8)

TMPL = {'imadw': None, 'stg': None, 'exit': None, 'nop': None}
first_stg = None
for off in range(0, len(SASS), 16):
    r = SASS[off:off+16]
    o = _op(r)
    if o == 0x825 and TMPL['imadw'] is None and r[4] != 0:  # IMAD.WIDE.U32 R,R,imm,R
        TMPL['imadw'] = r
    if o == 0x986:                                          # STG.E
        if TMPL['stg'] is None: TMPL['stg'] = r
        if first_stg is None: first_stg = off
    if o == 0x94d and TMPL['exit'] is None:                 # EXIT
        TMPL['exit'] = r
    if o == 0x918 and TMPL['nop'] is None:                  # NOP
        TMPL['nop'] = r
assert all(TMPL.values()), {k: (v is not None) for k, v in TMPL.items()}
print(f"[capture] first STG at 0x{first_stg:x}; templates ok", file=sys.stderr)

def _unpred(raw):                       # force guard -> PT (unpredicated)
    b = bytearray(raw); b[1] = (b[1] & 0x0f) | 0x70; return bytes(b)

# ============================ epilogue builder ============================
DUMP_N = NG                              # registers R0..R(NG-1) dumped
SCRATCH = (NG + 1) & ~1                   # even scratch pair for the address
STRIDE = ((DUMP_N * 4 + 15) & ~15)        # per-thread slot bytes (16-aligned)
NEW_NG = SCRATCH + 2

def epilogue():
    out = []
    nop = lambda: _patch_ctrl(TMPL['nop'], CTRL_WAIT)
    out += [nop(), nop()]                                   # drain memory pipes
    # IMAD.WIDE.U32 SCRATCH, R2, STRIDE, R6   (addr = gid*STRIDE + parents_base)
    iw = bytearray(_unpred(TMPL['imadw']))
    iw[2] = SCRATCH                                         # dest pair
    iw[3] = 2                                               # srcA = R2 (gid)
    iw[4:8] = struct.pack('<I', STRIDE)                    # imm
    # srcC (R6) lives in the high word of the template -> preserved
    out.append(_patch_ctrl(bytes(iw), CTRL_ALUW))
    out.append(nop())                                       # ALU GPR-latency gap
    for k in range(DUMP_N):                                 # STG.E [SCRATCH + 4k], Rk
        s = bytearray(_unpred(TMPL['stg']))
        s[3] = SCRATCH                                      # base
        s[4] = k                                            # data reg
        s[5:8] = struct.pack('<I', k * 4)[:3]              # 24-bit offset
        out.append(_patch_ctrl(bytes(s), CTRL_WAIT))
    out.append(_patch_ctrl(TMPL['exit'], CTRL_WAIT))
    return b''.join(out)

EPI = epilogue()

def build_cubin(T):
    """Truncate prefix at byte offset T, append dump epilogue, re-emit cubin."""
    new_sass = SASS[:T] + EPI
    eo = None
    for o in range(0, len(new_sass), 16):
        if _op(new_sass[o:o+16]) == 0x94d:
            eo = o; break
    nd = replace(DESC, sass_bytes=new_sass, num_gprs=NEW_NG, exit_offset=eo)
    return _ORIG_EMIT(nd)

# ============================ NOP-insertion confirmation builder ============================
# Insert a NOP after every TRUE-RAW pair of the 3 types that survived the
# _FORWARDING_SAFE_PAIRS force-gap test (they skip _enforce_gpr_latency).  Tests
# the hypothesis that these pairs need a 1-instruction ALU gap on SM_120.
LOP3 = {0x212, 0x812}; SHF = {0x819, 0x219}; IADD3 = {0x210}
def _is_pairtype(po, co):
    return ((po in LOP3 and co in SHF) or          # LOP3 -> SHF
            (po in SHF and co in IADD3) or          # SHF  -> IADD3
            (po in IADD3 and co in LOP3))           # IADD3-> LOP3

ALU_OPS_ALL = {0x210, 0x212, 0x812, 0x819, 0x219, 0x235, 0x224, 0x2a4, 0x810}
def build_nopfixed_cubin(lo=0x120, types=None, count=1, anyalu=False):
    """Full kernel + `count` NOPs after each qualifying true-RAW pair in
    [lo, end).  anyalu=True => any ALU->ALU true-RAW pair (broad latency test);
    else the 3 ARX pair-types (or `types` predicate).  Keeps the real store
    tail intact.  Returns (cubin, n_inserted, sample_sites)."""
    nop = _patch_ctrl(TMPL['nop'], CTRL_WAIT)
    out = bytearray(); n = 0; sites = []
    instrs = [SASS[o:o+16] for o in range(0, len(SASS), 16)]
    for i, r in enumerate(instrs):
        out += r
        off = i * 16
        if off < lo or i + 1 >= len(instrs):
            continue
        nxt = instrs[i + 1]
        po, co = _op(r), _op(nxt)
        if anyalu:
            ok = po in ALU_OPS_ALL and co in ALU_OPS_ALL
        elif types:
            ok = types(po, co)
        else:
            ok = _is_pairtype(po, co)
        if not ok:
            continue
        # true RAW: consumer reads a register the producer wrote
        if _get_dest_regs(r) & _get_src_regs(nxt):
            out += nop * count; n += 1
            if len(sites) < 8: sites.append((off, po, co))
    new_sass = bytes(out)
    eo = next(o for o in range(0, len(new_sass), 16) if _op(new_sass[o:o+16]) == 0x94d)
    nd = replace(DESC, sass_bytes=new_sass, exit_offset=eo)   # num_gprs unchanged
    return _ORIG_EMIT(nd), n, sites

# ============================ ctrl-slot confirmation builder ============================
# ptxas marks most ALU results wdep=0x3f (untracked, pipeline-latency) and rbar=0x1;
# ours slot-tracks EVERY ALU result on the single shared slot 0x3e (wdep=0x3e) with
# consumers waiting rbar=0x3 -> slot-counter ambiguity races.  Rewrite the pure-ALU
# compression region to ptxas's pattern and test if determinism returns.
ALU_OPS = {0x210, 0x212, 0x812, 0x819, 0x219, 0x235, 0x224, 0x2a4}
def _rd_ctrl(raw):
    return (int.from_bytes(raw[13:16], "little") >> 1) & 0x7FFFFF
def _mk_ctrl(stall, yld, wbar, rbar, wdep, misc):
    return (stall<<17)|(yld<<16)|(wbar<<15)|(rbar<<10)|(wdep<<4)|misc

def build_slotfixed_cubin(lo=0xbe0, hi=None, mode="slot"):
    """In [lo,hi) (pure-ALU compression), rewrite ALU ctrl.
    mode='slot': rbar 0x3->0x1, wdep 0x3e->0x3f (ptxas no-track pattern).
    mode='war' : set wbar=1 (write-after-read barrier) on every ALU op.
    Returns (cubin, n_patched)."""
    hi = hi or first_stg
    instrs = [bytearray(SASS[o:o+16]) for o in range(0, len(SASS), 16)]
    n = 0
    for i, r in enumerate(instrs):
        off = i * 16
        if off < lo or off >= hi or _op(r) not in ALU_OPS:
            continue
        c = _rd_ctrl(r)
        stall=(c>>17)&0x3f; yld=(c>>16)&1; wbar=(c>>15)&1
        rbar=(c>>10)&0x1f; wdep=(c>>4)&0x3f; misc=c&0xf
        if mode == "slot" and rbar == 0x3 and wdep == 0x3e:
            instrs[i] = bytearray(_patch_ctrl(bytes(r),
                _mk_ctrl(stall, yld, wbar, 0x1, 0x3f, misc))); n += 1
        elif mode == "war" and wbar == 0:
            instrs[i] = bytearray(_patch_ctrl(bytes(r),
                _mk_ctrl(max(stall,2), yld, 1, rbar, wdep, misc))); n += 1
    new_sass = b"".join(bytes(r) for r in instrs)
    eo = next(o for o in range(0, len(new_sass), 16) if _op(new_sass[o:o+16]) == 0x94d)
    return _ORIG_EMIT(replace(DESC, sass_bytes=new_sass, exit_offset=eo)), n

# ============================ latency-aware list scheduler (THE FIX) ============================
# The merkle race is: ours emits dependent ALU back-to-back (0 real-instr gap);
# the SM_120 FXU->FXU RAW window needs ~3 real instructions of independent work
# (ptxas modal gap=3).  NOPs don't cover it (don't occupy FXU).  This list
# scheduler reorders the pure-ALU compression region to interleave independent
# instructions, filling the latency window with real work.
# Predicate-aware I/O: model P0..P6 as pseudo-registers (PBASE+p) so the
# scheduler tracks carry/guard deps.  For merkle only P0 is used (compression
# carries are dead P0 writes; store-guard uses P0 for live carry + guard).
PBASE = 300
_IADD3_OPS = {0x210, 0x810}                 # IADD3 / IADD3.X (carry-out to P0)
def _pred_io(raw):
    op = _op(raw); pw = set(); pr = set()
    g = (raw[1] >> 4) & 0xF                  # guard predicate read
    if g != 0x7: pr.add(PBASE + (g & 0x7))
    if op in _IADD3_OPS:                      # carry through P0 (conservative)
        pw.add(PBASE + 0); pr.add(PBASE + 0)
    if op == 0xc0c:            pw.add(PBASE + raw[2])              # ISETP R-UR
    elif op in (0x20c, 0x80c): pw.add(PBASE + ((raw[10] >> 1) & 7))  # ISETP R-R/IMM
    elif op in (0xc0b, 0x20b): pw.add(PBASE + (raw[9] & 7))       # FSETP
    return pw, pr

def _list_schedule(raws, L=4, pred_aware=False):
    """raws: list of 16-byte instrs.  Returns reordered list with NOPs only on
    genuine latency stalls.  RAW edges carry latency L; WAR/WAW carry 1.
    pred_aware adds P0..P6 carry/guard deps (needed for the store-guard chain)."""
    n = len(raws)
    dest = [set(_get_dest_regs(r)) for r in raws]
    src  = [set(_get_src_regs(r)) for r in raws]
    if pred_aware:
        for i, r in enumerate(raws):
            pw, pr = _pred_io(r); dest[i] |= pw; src[i] |= pr
    succs = [[] for _ in range(n)]; npred = [0]*n
    for i in range(n):
        for j in range(i):
            lat = 0
            if dest[j] & src[i]:                       lat = max(lat, L)   # RAW
            if (src[j] & dest[i]) or (dest[j] & dest[i]): lat = max(lat, 1) # WAR/WAW
            if lat:
                succs[j].append((i, lat)); npred[i] += 1
    cp = [1]*n                                          # critical-path priority
    for i in range(n-1, -1, -1):
        if succs[i]: cp[i] = 1 + max(lat + cp[k] for (k, lat) in succs[i])
    earliest = [0]*n; rem = npred[:]
    ready = {i for i in range(n) if rem[i] == 0}
    out, cycle, done = [], 0, 0
    while done < n:
        avail = [i for i in ready if earliest[i] <= cycle]
        if avail:
            pick = max(avail, key=lambda i: (cp[i], -i))
            ready.discard(pick); out.append(raws[pick]); done += 1
            for (k, lat) in succs[pick]:
                earliest[k] = max(earliest[k], cycle + lat)
                rem[k] -= 1
                if rem[k] == 0: ready.add(k)
        else:
            out.append(TMPL['nop'])                            # stall filler (valid NOP)
        cycle += 1
        if cycle > 20*n: raise RuntimeError("scheduler stuck")
    return out

# ---- register renaming: SSA-rename the pure-ALU compression to break false
# WAR/WAW deps, list-schedule the resulting RAW-only DAG (max parallelism),
# then linear-scan allocate physical regs + fixup MOVs for live-outs. ----
# layout: opcode -> (dest_byte, [gpr_src_bytes])  (RZ=255 not renamed; imm bytes excluded)
_LAYOUT = {0x819:(2,[3,8]), 0x212:(2,[3,4,8]), 0x210:(2,[3,4,8]),
           0x235:(2,[3,4]), 0x835:(2,[3]), 0x812:(2,[3,8]), 0x202:(2,[4])}
OUT_REGS = [36, 23, 20, 25, 34, 29, 15, 5]       # 8 hash words the tail stores

def _movrr_tmpl():
    for o in range(0, len(SASS), 16):
        if _op(SASS[o:o+16]) == 0x202: return SASS[o:o+16]
    raise RuntimeError("no MOV R-R template")

def _rename_schedule(region, L=4):
    """region: list of real (non-NOP) pure-ALU raws. Returns scheduled+renamed
    raws (+ stall NOPs + live-out fixup MOVs)."""
    n = len(region)
    lay = [_LAYOUT[_op(r)] for r in region]
    # --- SSA rename: assign each def a value-id; reads map to current value ---
    cur = {}                       # orig phys reg -> value id
    vprod = []                     # value id -> producer instr idx (or None=live-in)
    vphys_in = []                  # value id -> pinned phys (live-in) or None
    isrc = [[] for _ in range(n)]  # instr -> [value ids it reads]
    idst = [0]*n                   # instr -> value id it writes
    def livein(reg):
        v = len(vprod); vprod.append(None); vphys_in.append(reg); cur[reg]=v; return v
    for i, r in enumerate(region):
        db, sbs = lay[i]
        for b in sbs:
            rg = r[b]
            if rg == 255: continue                  # RZ
            if rg not in cur: livein(rg)
            isrc[i].append(cur[rg])
        v = len(vprod); vprod.append(i); vphys_in.append(None)
        idst[i] = v; cur[r[db]] = v
    nv = len(vprod)
    out_vals = {cur[rg]: rg for rg in OUT_REGS if rg in cur}   # final value -> required reg
    # readers per value (for liveness); live-outs get +1 phantom reader
    readers = [[] for _ in range(nv)]
    for i in range(n):
        for v in isrc[i]: readers[v].append(i)
    # --- DAG: RAW edges producer->reader (latency L) ---
    succ = [[] for _ in range(n)]; npred=[0]*n
    for i in range(n):
        ps=set()
        for v in isrc[i]:
            if vprod[v] is not None: ps.add(vprod[v])
        for p in ps:
            succ[p].append(i); npred[i]+=1
    cp=[1]*n
    for i in range(n-1,-1,-1):
        if succ[i]: cp[i]=1+max(L+cp[k] for k in succ[i])
    # --- list schedule ---
    earliest=[0]*n; rem=npred[:]; ready={i for i in range(n) if rem[i]==0}
    order=[]; cyc=0; done=0; sched_pos={}
    while done<n:
        av=[i for i in ready if earliest[i]<=cyc]
        if av:
            pk=max(av,key=lambda i:(cp[i],-i)); ready.discard(pk)
            sched_pos[pk]=len(order); order.append(pk); done+=1
            for k in succ[pk]:
                earliest[k]=max(earliest[k],cyc+L); rem[k]-=1
                if rem[k]==0: ready.add(k)
        else: order.append(None)            # stall
        cyc+=1
        if cyc>40*n: raise RuntimeError("rename-sched stuck")
    # --- linear-scan physical allocation over scheduled order ---
    POOL=list(range(200, 41, -1))           # fresh regs R42..R199 (reuse via free list)
    phys={}                                  # value id -> phys reg
    for v in range(nv):
        if vphys_in[v] is not None: phys[v]=vphys_in[v]   # live-in pinned
    # remaining uses per value in scheduled order; free phys when hits 0 (unless live-out)
    remuse=[len(readers[v]) for v in range(nv)]
    out=[]
    movrr=_movrr_tmpl(); maxreg=[41]
    def setregs(raw, dv, svs):
        b=bytearray(raw); db,sbs=_LAYOUT[_op(raw)]
        b[db]=dv; maxreg[0]=max(maxreg[0],dv)
        si=0
        for bb in sbs:
            if raw[bb]==255: continue
            b[bb]=svs[si]; maxreg[0]=max(maxreg[0],svs[si]); si+=1
        return bytes(b)
    for slot in order:
        if slot is None: out.append(TMPL['nop']); continue
        i=slot
        svs=[phys[v] for v in isrc[i]]
        dv=idst[i]
        if dv in out_vals: ph=POOL.pop()      # live-out: alloc, keep (fixup later)
        elif not readers[dv] and dv not in out_vals: ph=POOL.pop()  # dead def (rare)
        else: ph=POOL.pop()
        phys[dv]=ph
        out.append(setregs(region[i], ph, svs))
        for v in isrc[i]:                      # free sources whose last use is now
            remuse[v]-=1
            if remuse[v]==0 and v not in out_vals and vphys_in[v] is None:
                POOL.append(phys[v])
    # --- live-out fixup: MOV required_reg <- phys(final value) ---
    for v,reg in out_vals.items():
        b=bytearray(movrr); b[2]=reg; b[4]=phys[v]; maxreg[0]=max(maxreg[0],reg,phys[v])
        out.append(bytes(b))
    return out, maxreg[0]

def build_renamed_cubin(lo=0xbe0, hi=0x4860, L=4, sg=True):
    """Rename+schedule compression [0xbe0,0x4860); then (if sg) predicate-aware
    schedule the store-guard [0x4860,0x48d0); splice + emit."""
    NOP=_op(TMPL['nop'])
    comp=[SASS[o:o+16] for o in range(0xbe0,0x4860,16) if _op(SASS[o:o+16])!=NOP]
    rsched,maxreg=_rename_schedule(comp,L=L)
    if sg:
        sgreg=[SASS[o:o+16] for o in range(0x4860,0x48d0,16) if _op(SASS[o:o+16])!=NOP]
        sgsched=_list_schedule(sgreg,L=L,pred_aware=True)
        mid=b"".join(rsched)+b"".join(sgsched)
        post=SASS[0x48d0:]
    else:
        mid=b"".join(rsched); post=SASS[0x4860:]
    new_sass=SASS[:0xbe0]+mid+post
    eo=next(o for o in range(0,len(new_sass),16) if _op(new_sass[o:o+16])==0x94d)
    ng=max(NEW_NG, maxreg+2)
    return _ORIG_EMIT(replace(DESC,sass_bytes=new_sass,num_gprs=ng,exit_offset=eo)), len(rsched), sum(1 for r in rsched if _op(r)==NOP), maxreg

def build_scheduled_cubin(lo=0xbe0, hi=0x4860, L=4, pred_aware=False):
    """Re-list-schedule region [lo,hi); keep original ctrl on each real instr
    (ALU ctrl is inert per slotfix); NOP only on genuine stalls.
    pred_aware + extended hi (0x48d0) also covers the store-guard carry chain."""
    pre = SASS[:lo]; post = SASS[hi:]
    region = [SASS[o:o+16] for o in range(lo, hi, 16)]
    NOP = _op(TMPL['nop'])
    real = [r for r in region if _op(r) != NOP]          # strip existing padding
    sched = _list_schedule(real, L=L, pred_aware=pred_aware)
    new_sass = pre + b"".join(sched) + post
    eo = next(o for o in range(0, len(new_sass), 16) if _op(new_sass[o:o+16]) == 0x94d)
    nins = len(sched); nnops = sum(1 for r in sched if _op(r) == NOP)
    return _ORIG_EMIT(replace(DESC, sass_bytes=new_sass, exit_offset=eo)), len(real), nins, nnops

# ============================ GPU launch (K in-process runs) ============================
# In-process repeats keep device pointers STABLE across the K launches (verified:
# parents ptr constant within a process), so pointer-carrying registers do NOT
# show as non-det — only the timing race does.  A re-zeroed double-run still
# exposes the race (merkle_inproc_test.py), so this is reliable here.
_GPU = {}                                        # one shared context + buffers
def _gpu():
    if not _GPU:
        ctx = workbench.CUDAContext()
        cd = ctx.alloc(SZ); ctx.copy_to(cd, children)
        pd = ctx.alloc(SZ)
        _GPU.update(ctx=ctx, cd=cd, pd=pd)
    return _GPU['ctx'], _GPU['cd'], _GPU['pd']

def _gpu_reset():                                # tear down a poisoned context
    try: _GPU['ctx'].close()
    except Exception: pass
    _GPU.clear()

def _err(cuda, code):
    s = ctypes.c_char_p()
    cuda.cuGetErrorString(code, ctypes.byref(s))
    return f"{code}:{(s.value or b'?').decode(errors='replace')}"

def gpu_dump_k(cubin_bytes, K):
    ctx, cd, pd = _gpu()
    if not ctx.load(cubin_bytes):
        return [('FAIL', 'load')] * K
    f = ctx.get_func(KERNEL)
    used = 256 * STRIDE
    vals = [ctypes.c_uint64(cd), ctypes.c_uint64(N), ctypes.c_uint64(pd),
            ctypes.c_uint64(N), ctypes.c_uint64(N)]
    a = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    runs = []
    for _ in range(K):
        ctx.memset_d8(pd, 0, used)
        le = ctx.cuda.cuLaunchKernel(f, (N + 255)//256, 1, 1, 256, 1, 1, 0, None, a, None)
        se = ctx.sync()
        if le or se:                              # sticky error -> reset context
            runs.append(('FAIL', _err(ctx.cuda, le or se)))
            _gpu_reset(); ctx, cd, pd = _gpu()
            if not ctx.load(cubin_bytes): return runs + [('FAIL','reload')]*(K-len(runs))
            f = ctx.get_func(KERNEL)
            continue
        raw = bytes(ctx.copy_from(pd, STRIDE)[0:DUMP_N*4])   # gid=0 slot
        runs.append([struct.unpack_from('<I', raw, 4*k)[0] for k in range(DUMP_N)])
    return runs

def gpu_output_k(cubin_bytes, K):
    """Launch the REAL kernel K times; return list of parent0[0:32] hex (or FAIL)."""
    ctx, cd, pd = _gpu()
    if not ctx.load(cubin_bytes):
        return ['FAIL:load'] * K
    f = ctx.get_func(KERNEL)
    vals = [ctypes.c_uint64(cd), ctypes.c_uint64(N), ctypes.c_uint64(pd),
            ctypes.c_uint64(N), ctypes.c_uint64(N)]
    a = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    outs = []
    for _ in range(K):
        ctx.memset_d8(pd, 0, 64)
        le = ctx.cuda.cuLaunchKernel(f, (N + 255)//256, 1, 1, 256, 1, 1, 0, None, a, None)
        se = ctx.sync()
        if le or se:
            outs.append('FAIL:' + _err(ctx.cuda, le or se)); _gpu_reset()
            ctx, cd, pd = _gpu()
            if not ctx.load(cubin_bytes): return outs + ['FAIL:reload']*(K-len(outs))
            f = ctx.get_func(KERNEL); continue
        outs.append(bytes(ctx.copy_from(pd, 64)[0:32]).hex())
    return outs

# ============================ emulator oracle ============================
def emu_regfile(T, instrs):
    """Run the emulator on the truncated path; return (regvals, defined_set).
    defined_set = GPRs actually WRITTEN by offset T — only these carry a
    meaningful ground-truth value; un-written regs hold non-det GPU garbage
    that must be ignored (cfg_live invalid-path red herring)."""
    s = E.new_state(); s.ntid = (256,1,1); s.ctaid=(0,0,0); s.tid=(0,0,0)
    cb = bytearray(0x400)
    struct.pack_into("<I", cb, 0x360, 256)
    struct.pack_into("<Q", cb, 0x380, CHILD_BASE)
    struct.pack_into("<Q", cb, 0x388, N)
    struct.pack_into("<Q", cb, 0x390, PAR_BASE)
    struct.pack_into("<Q", cb, 0x398, N)
    struct.pack_into("<Q", cb, 0x3a0, N)
    s.cbank = bytes(cb)
    s.globals[CHILD_BASE] = bytearray(children)
    s.globals[PAR_BASE] = bytearray(SZ)
    defined = set()
    _orig_wg = E.write_gpr
    def _wg(state, idx, val):
        if idx != 255: defined.add(idx)
        _orig_wg(state, idx, val)
    E.write_gpr = _wg
    try:
        trunc = [ins for ins in instrs if ins.pc < T]
        E.run(s, trunc, max_steps=200000)
    finally:
        E.write_gpr = _orig_wg
    return ([s.gprs[k] & 0xFFFFFFFF for k in range(DUMP_N)], defined)

# ============================ commands ============================
def _ptr_like(v):                                # emu value that is a synthetic pointer
    return v >= 0x10000000

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if not os.path.exists("/tmp/ours_merkle.cubin"):
        open("/tmp/ours_merkle.cubin", "wb").write(_ORIG_EMIT(DESC))
    instrs = E.decode_kernel("/tmp/ours_merkle.cubin", KERNEL)

    if cmd == "rsched":                          # FULL FIX: rename + global schedule
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        L = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        EMU_GT = "5377e4ff957bda4d4535f4879876b71a61056c4cec31e78397c66ec47a86a130"
        cubin, nslots, nnops, maxreg = build_renamed_cubin(L=L)
        print(f"renamed+scheduled compression: {nslots} slots, {nnops} stall-NOPs, max GPR R{maxreg}")
        open("/tmp/rsched.cubin", "wb").write(cubin)
        si = E.decode_kernel("/tmp/rsched.cubin", KERNEL)
        s = E.new_state(); s.ntid=(256,1,1); s.ctaid=(0,0,0); s.tid=(0,0,0)
        cb = bytearray(0x400)
        for off,val,w in [(0x360,256,4),(0x380,CHILD_BASE,8),(0x388,N,8),(0x390,PAR_BASE,8),(0x398,N,8),(0x3a0,N,8)]:
            struct.pack_into("<I" if w==4 else "<Q", cb, off, val)
        s.cbank=bytes(cb); s.globals[CHILD_BASE]=bytearray(children); s.globals[PAR_BASE]=bytearray(SZ)
        E.run(s, si, max_steps=500000)
        emu_out = bytes(s.globals[PAR_BASE][0:32]).hex()
        print(f"emulator: {emu_out[:48]}  {'== GT (semantics OK)' if emu_out==EMU_GT else '!! BROKEN'}")
        outs = gpu_output_k(cubin, K)
        t = {}
        for o in outs: t[o] = t.get(o, 0) + 1
        verd = "DETERMINISTIC" if len(t)==1 else f"NON-DET ({len(t)} distinct)"
        gt = "  == GT (CORRECT+DETERMINISTIC)" if (len(t)==1 and EMU_GT in t) else ""
        print(f"GPU: {verd}{gt} over {K} runs")
        for v,c in sorted(t.items(), key=lambda kv:-kv[1])[:5]:
            print(f"  {c:2d}x  {v[:48]}{'  <-GT' if v==EMU_GT else ''}")
        return

    if cmd in ("sched", "gsched"):               # THE FIX: list-schedule the ALU region
        # gsched = global: extended region [0xbe0,0x48d0) + predicate-aware (store-guard)
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        L = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        EMU_GT = "5377e4ff957bda4d4535f4879876b71a61056c4cec31e78397c66ec47a86a130"
        if cmd == "gsched":
            cubin, nreal, nins, nnops = build_scheduled_cubin(lo=0xbe0, hi=0x48d0, L=L, pred_aware=True)
        else:
            cubin, nreal, nins, nnops = build_scheduled_cubin(L=L)
        print(f"list-scheduled region: {nreal} real instrs -> {nins} slots ({nnops} stall-NOPs), L={L}")
        open("/tmp/sched.cubin", "wb").write(cubin)
        ev, eo2, _ = __import__("merkle_emu").emulate("/tmp/sched.cubin") if False else (None,None,None)
        # emulate the scheduled stream to verify semantics preserved
        si = E.decode_kernel("/tmp/sched.cubin", KERNEL)
        s = E.new_state(); s.ntid=(256,1,1); s.ctaid=(0,0,0); s.tid=(0,0,0)
        cb = bytearray(0x400)
        for off,val,w in [(0x360,256,4),(0x380,CHILD_BASE,8),(0x388,N,8),(0x390,PAR_BASE,8),(0x398,N,8),(0x3a0,N,8)]:
            struct.pack_into("<I" if w==4 else "<Q", cb, off, val)
        s.cbank=bytes(cb); s.globals[CHILD_BASE]=bytearray(children); s.globals[PAR_BASE]=bytearray(SZ)
        E.run(s, si, max_steps=500000)
        emu_out = bytes(s.globals[PAR_BASE][0:32]).hex()
        print(f"emulator on scheduled stream: {emu_out[:48]}  {'== GT (semantics preserved)' if emu_out==EMU_GT else '!! SEMANTICS BROKEN'}")
        outs = gpu_output_k(cubin, K)
        t = {}
        for o in outs: t[o] = t.get(o, 0) + 1
        verd = "DETERMINISTIC" if len(t) == 1 else f"NON-DET ({len(t)} distinct)"
        gt = "  == emu GROUND TRUTH (CORRECT + DETERMINISTIC!)" if (len(t)==1 and EMU_GT in t) else ""
        print(f"scheduled: {verd}{gt} over {K} runs")
        for v, c in sorted(t.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {c:2d}x  {v[:48]}{'  <-GT' if v==EMU_GT else ''}")
        return

    if cmd in ("slotfix", "warfix"):             # confirmation: rewrite ALU ctrl, re-run
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        EMU_GT = "5377e4ff957bda4d4535f4879876b71a61056c4cec31e78397c66ec47a86a130"
        mode = "slot" if cmd == "slotfix" else "war"
        cubin, n = build_slotfixed_cubin(mode=mode)
        print(f"patched {n} ARX ALU ops (mode={mode})")
        outs = gpu_output_k(cubin, K)
        t = {}
        for o in outs: t[o] = t.get(o, 0) + 1
        verd = "DETERMINISTIC" if len(t) == 1 else f"NON-DET ({len(t)} distinct)"
        gt = "  == emu GROUND TRUTH (CORRECT!)" if (len(t)==1 and EMU_GT in t) else ""
        print(f"slotfixed: {verd}{gt} over {K} runs")
        for v, c in sorted(t.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {c:2d}x  {v[:48]}{'  <-GT' if v==EMU_GT else ''}")
        return

    if cmd == "nopfix":                          # confirmation: insert gaps, re-run real output
        # nopfix [K] [count] [anyalu]
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        anyalu = len(sys.argv) > 4 and sys.argv[4] == "anyalu"
        EMU_GT = "5377e4ff957bda4d4535f4879876b71a61056c4cec31e78397c66ec47a86a130"
        # baseline (unmodified) for contrast
        base = gpu_output_k(_ORIG_EMIT(DESC), K)
        cubin, n, sites = build_nopfixed_cubin(count=count, anyalu=anyalu)
        print(f"inserted {count} NOP(s) x {n} {'ANY-ALU' if anyalu else 'ARX'} true-RAW pairs")
        print(f"  sample sites (off, prod_op, cons_op): "
              + ", ".join(f"0x{o:x}:{p:#05x}->{c:#05x}" for o, p, c in sites))
        outs = gpu_output_k(cubin, K)
        for tag, res in (("baseline", base), ("nopfixed", outs)):
            t = {}
            for o in res: t[o] = t.get(o, 0) + 1
            verd = "DETERMINISTIC" if len(t) == 1 else f"NON-DET ({len(t)} distinct)"
            gt = "  == emu GT" if (len(t) == 1 and EMU_GT in t) else ""
            print(f"\n{tag}: {verd}{gt} over {K} runs")
            for v, c in sorted(t.items(), key=lambda kv: -kv[1])[:6]:
                print(f"  {c:2d}x  {v[:48]}")
        return

    if cmd in ("at", "validate"):
        T = int(sys.argv[2], 16) if cmd == "at" else 0x120
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        runs = gpu_dump_k(build_cubin(T), K)
        emu, defined = emu_regfile(T, instrs)
        report(T, runs, emu, defined)
        return

    if cmd == "rscan":                           # windowed refine: rscan lo hi [stride] [K]
        lo = int(sys.argv[2], 16); hi = int(sys.argv[3], 16)
        stride = int(sys.argv[4], 16) if len(sys.argv) > 4 else 0x10
        K = int(sys.argv[5]) if len(sys.argv) > 5 else 6
        T, first = lo, None
        while T <= hi:
            runs = gpu_dump_k(build_cubin(T), K)
            emu, defined = emu_regfile(T, instrs)
            nd, dv = _race_regs(runs, emu, defined)
            mark = "  <-- FIRST NONDET" if (nd and first is None) else ""
            if nd and first is None: first = T
            print(f"T=0x{T:04x}  def={len(defined):2d}  "
                  f"NONDET={['R%d'%k for k in nd]}{mark}", flush=True)
            T += stride
        if first:
            print(f"\n>>> first racing instruction is at 0x{first-0x10:x} "
                  f"(its result is non-det at checkpoint 0x{first:x})")
        return

    if cmd == "scan":
        stride = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0x200
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        T, hi, prev_clean = 0x120, first_stg, None
        while T < hi:
            runs = gpu_dump_k(build_cubin(T), K)
            emu, defined = emu_regfile(T, instrs)
            verdict = classify(runs, emu, defined)
            print(f"T=0x{T:04x}  defined={len(defined):2d}  {verdict}", flush=True)
            if verdict.startswith("NONDET"):
                print(f"\n>>> race first manifest in (0x{prev_clean:x}, 0x{T:x}]"
                      if prev_clean else f"\n>>> race by 0x{T:x}")
                report(T, runs, emu, defined)
                return
            prev_clean = T
            T += stride
        print("no race found in straight-line region")
        return

def _race_regs(runs, emu, defined):
    """Return (nondet, diverge) register lists over DEFINED regs; pointer-valued
    emu regs are excluded from DET-DIVERGE (GPU real ptr != emu synthetic)."""
    good = [r for r in runs if isinstance(r, list)]
    nondet, diverge = [], []
    for k in sorted(defined):
        if k >= DUMP_N: continue
        seen = {r[k] for r in good}
        if len(seen) > 1:
            nondet.append(k)
        elif not _ptr_like(emu[k]) and next(iter(seen)) != emu[k]:
            diverge.append(k)
    return nondet, diverge

def classify(runs, emu, defined):
    fails = [r for r in runs if not isinstance(r, list)]
    if fails: return f"LAUNCHFAIL ({len(fails)}/{len(runs)})"
    nondet, diverge = _race_regs(runs, emu, defined)
    if nondet:  return f"NONDET regs={['R%d'%k for k in nondet]}"
    if diverge: return f"DET-DIVERGE regs={['R%d'%k for k in diverge]}"
    return "DET-MATCH"

def report(T, runs, emu, defined):
    print(f"--- checkpoint T=0x{T:x}  ({T//16} instrs; {len(defined)} regs written) ---")
    print(f"verdict: {classify(runs, emu, defined)}")
    good = [r for r in runs if isinstance(r, list)]
    if len(good) < len(runs):
        print(f"  ({len(runs)-len(good)} launch failures: {[r for r in runs if not isinstance(r,list)][:2]})")
    print("reg   emu       " + "  ".join(f"run{i}" for i in range(len(good))))
    for k in sorted(defined):
        if k >= DUMP_N: continue
        cols = [f"{r[k]:08x}" for r in good]
        nondet = len(set(cols)) > 1
        match = all(r[k] == emu[k] for r in good)
        tag = "ptr" if _ptr_like(emu[k]) else ""
        flag = "  <<NONDET" if nondet else ("" if match else f"  <<DIVERGE {tag}")
        print(f"R{k:<3d} {emu[k]:08x}  " + "  ".join(cols) + flag)

if __name__ == "__main__":
    main()
