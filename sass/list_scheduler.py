"""
sass/list_scheduler.py — latency-aware list scheduler + register renaming.

Fixes the SM_120 FXU->FXU RAW hazard class: dependent ALU ops emitted
back-to-back (0 real-instruction gap) read a producer's result before the
fixed ALU pipeline has written it back, producing NON-DETERMINISTIC output
that NOP padding / scoreboard ctrl cannot fix (only real independent
instructions interleaved into the latency window work — see merkle_hash_nodes
investigation 2026-06-16).  ptxas covers this by interleaving ~3 independent
instructions between dependent ALU ops (modal gap=3) and renaming registers to
expose the parallelism.  This pass does the same.

Scope (conservative, gated by OPENPTXAS_LIST_SCHED):
  * Operates on maximal PURE-ALU runs only — runs bounded by memory
    (LDG/STG/ATOM/LDS/STS), control (BRA/EXIT/BSYNC/BAR/MEMBAR/DEPBAR),
    UR-writers (LDCU/S2R/S2UR/REDUX) and SHFL/VOTE.  Those ops are FENCES:
    never reordered across, never inside a scheduled run.
  * Within a run: list-schedule on a dependency DAG (RAW latency L; WAR/WAW/
    pred/carry latency 1) and rename simple-32-bit-ALU intermediate values to
    break false WAR/WAW deps.  Live-in, live-out, pair (IMAD.WIDE/IADD.64) and
    predicate-involved values keep their original registers (no fixup MOVs;
    downstream/cross-run readers are untouched).
  * Carry/guard predicates modeled conservatively as a single P0..P6 pseudo-
    resource so the carry chain (IADD3->IADD3.X->ISETP->@P STG) stays correct.

Ctrl words are reassigned downstream by sass.scoreboard.assign_ctrl(), so this
pass only needs to produce a correct ORDER (with stall NOPs where no
independent work is ready).
"""
from __future__ import annotations
import os
import bisect as _bisect
from sass.isel import SassInstr
from sass.scoreboard import _get_opcode, _get_src_regs, _get_dest_regs
from sass.encoding.sm_120_opcodes import encode_nop as _enc_nop

# Simple 32-bit ALU opcodes we can re-encode (dest=b2; GPR srcs at listed bytes).
_LAYOUT = {
    0x819: (2, (3, 8)),      # SHF (b4 = imm shift)
    0x212: (2, (3, 4, 8)),   # LOP3.LUT R-R-R
    0x812: (2, (3, 8)),      # LOP3.LUT R-imm
    0x210: (2, (3, 4, 8)),   # IADD3 R-R-R
    0x810: (2, (3, 8)),      # IADD3 R-imm
    0x235: (2, (3, 4)),      # IADD R-R
    0x835: (2, (3,)),        # IADD R-imm
}
# Fences: never inside a run / never reorder across.
_FENCE = (
    {0x981, 0xf60, 0xf63, 0xf66, 0xf6f, 0xf99}           # LDG/TEX family
    | {0x986, 0xf9d, 0x988, 0x388, 0x387}                # STG/STS
    | {0x3a9, 0x3a8, 0x3aa}                              # ATOMG-ish (conservative)
    | {0x947, 0x94d, 0xb1d, 0x941, 0x992, 0x91a, 0x9af} # BRA/EXIT/BAR/BSYNC/MEMBAR/DEPBAR
    | {0x919, 0x9c3, 0x3c4}                              # S2R/S2UR/REDUX (UR/special writes)
    | {0x589, 0xf89, 0x989, 0x806}                      # SHFL/VOTE
    | {0xfae}                                            # LDGSTS
)
_NOP = 0x918
_PBASE = 1000                                            # predicate pseudo-reg base
_IADD3 = {0x210, 0x810}                                  # carry-out to P0 (conservative)
# (producer_op, consumer_op) pairs EMPIRICALLY VALIDATED to forward at gap 0 on
# SM_120 (consumer can read the producer's result back-to-back, no real-instr
# gap needed).  These get RAW latency 1 (not L) so the scheduler doesn't waste
# spacing, and the hazard-gate ignores them.  SOUND: only proven-safe pairs go
# here; every other FXU->FXU RAW defaults to L (treated as racing).
#   (IADD3, SHF): validated via m31_scale (gap-0 IADD3->SHF, 150/150 deterministic).
_FORW_SAFE = {(0x210, 0x819)}
_ISETP = {0xc0c: 'ur', 0x20c: 'rr', 0x80c: 'rr', 0xc0b: 'f', 0x20b: 'f'}
_L_DEFAULT = 4
_COOL = int(os.environ.get("OPENPTXAS_LIST_SCHED_COOL", "24"))  # reg-reuse cooldown (positions);
# must exceed the ALU WRITE-BACK latency (longer than the forwarding gap L) so a reused reg's
# prior occupant has retired.  8 was marginal (~1/300 illegal-addr); 16/24/32 clean at K=300.
_MIN_RUN = int(os.environ.get("OPENPTXAS_LIST_SCHED_MINRUN", "16"))
_RENAME = not os.environ.get("OPENPTXAS_LIST_SCHED_NORENAME")   # on by default; reg-reuse hazard
# fixed by the _COOL cooldown (a freed reg waits _COOL positions before reuse so the prior
# occupant's write-back completes).  Validated 300/300 == GT on merkle_nodes.


def _pred_io(raw):
    """(pred_writes, pred_reads) as pseudo-reg sets (conservative P0 carry)."""
    op = _get_opcode(raw); pw = set(); pr = set()
    g = (raw[1] >> 4) & 0xF
    if g != 0x7:
        pr.add(_PBASE + (g & 7))                         # @Px / @!Px guard read
    if op in _IADD3:                                     # precise carry decode:
        b9, b10 = raw[9], raw[10]                        # byte9=0xe0 plain / 0xe4 .X
        if b9 == 0xe4:                                   # IADD3.X reads carry-in (P0/P1)
            pr.add(_PBASE + (0 if b10 == 0x7f else 1))
        else:                                            # plain IADD3 may write carry-out
            po = (b10 >> 1) & 7
            if po != 7: pw.add(_PBASE + po)              # 0xf1->P0, 0xf3->P1, 0xff->none
    k = _ISETP.get(op)
    if k == 'ur':   pw.add(_PBASE + raw[2])
    elif k == 'rr': pw.add(_PBASE + ((raw[10] >> 1) & 7))
    elif k == 'f':  pw.add(_PBASE + (raw[9] & 7))
    return pw, pr


def _io(raw):
    """Full (writes, reads) including GPR (pairs handled by scoreboard helpers)
    and predicate pseudo-regs."""
    d = set(_get_dest_regs(raw)); s = set(_get_src_regs(raw))
    pw, pr = _pred_io(raw)
    return (d | pw), (s | pr)


def _runs(instrs, lo):
    """Yield (start, end) index ranges of maximal pure-ALU runs in instrs[lo:]
    (excluding fences and NOPs at the edges)."""
    i = lo; n = len(instrs)
    while i < n:
        op = _get_opcode(instrs[i].raw)
        if op in _FENCE or op == _NOP:
            i += 1; continue
        j = i
        while j < n:
            o = _get_opcode(instrs[j].raw)
            if o in _FENCE:
                break
            j += 1
        # run is instrs[i:j]; only reschedule large ALU blocks (where the
        # FXU-latency hazard concentrates) — small address-arith runs between
        # loads are left untouched to minimize blast radius.
        if j - i >= _MIN_RUN:
            yield (i, j)
        i = j


def schedule_and_rename(instrs, body_start, L=_L_DEFAULT):
    """Top-level: schedule+rename pure-ALU runs in instrs[body_start:].
    Returns a new SassInstr list.  Pure function; ctrl reassigned downstream."""
    # live-out reg set per run is computed globally: a reg is live-out of a run
    # if it is READ somewhere after the run's end before being redefined.
    out = list(instrs)
    runs = list(_runs(out, body_start))
    # CFG-aware precise liveness (opt-in OPENPTXAS_CFG_LIVENESS): the true set of
    # registers live across each run (cfg.live_after_index, incl. back-edges)
    # replaces the conservative used-anywhere-outside-run set. Precomputed on the
    # ORIGINAL stream -- renaming preserves the live-register SETS, so the sets stay
    # valid across splicing.
    _cfg_live = {}
    if os.environ.get('OPENPTXAS_CFG_LIVENESS') and runs:
        try:
            from sass.cfg import build_cfg, compute_liveness, live_after_index
            _cfg = build_cfg(out)
            _li, _lo = compute_liveness(out, _cfg)
            for (_s, _e) in runs:
                _cfg_live[(_s, _e)] = live_after_index(out, _cfg, _lo, _e - 1)
        except Exception:
            _cfg_live = {}
    # process runs back-to-front so index ranges stay valid as we splice
    for (s, e) in reversed(runs):
        # HAZARD-GATE: only reschedule runs that actually contain the FXU->FXU
        # latency hazard (a RAW-dependent FXU pair at < L real-instruction gap).
        # Clean, already-well-spaced runs (e.g. m31_scale) are left untouched so
        # the pass never bloats non-racing kernels.
        if not _has_fxu_hazard(out[s:e], L):
            continue
        if (s, e) in _cfg_live:
            # CFG-precise: regs live across the run gate BOTH the rename pool
            # (don't clobber a live reg) and visout (don't rename a live value).
            reserved = _cfg_live[(s, e)]
            liveout = _cfg_live[(s, e)]
        else:
            liveout = _liveout_after(out, e)
            # regs touched anywhere OUTSIDE this run: CFG-sound superset of
            # everything live across the run on any path (incl. back-edges).
            reserved = set()
            for _k in range(len(out)):
                if s <= _k < e:
                    continue
                _r = out[_k].raw
                if _get_opcode(_r) == _NOP:
                    continue
                for _x in _get_src_regs(_r):
                    if _x < 255:
                        reserved.add(_x)
                for _x in _get_dest_regs(_r):
                    if _x < 255:
                        reserved.add(_x)
        new = _sched_rename_run(out[s:e], liveout, L, reserved)
        out[s:e] = new
    return out


def _has_fxu_hazard(run, L):
    """True if the run has a GAP-0 (back-to-back, 0 real instrs between)
    RAW-dependent FXU->FXU pair that isn't forwarding-safe — the SM_120
    write-back-latency hazard this pass fixes.  ours emits the buggy pairs at
    gap-0-real (NOP padding doesn't count); genuinely-spaced pairs (a real
    instr between, e.g. m31_scale's gap-1) and validated forwarding pairs
    (_FORW_SAFE) are safe and left alone — so clean kernels aren't rescheduled."""
    real = [si for si in run if _get_opcode(si.raw) != _NOP]
    n = len(real)
    opc = [_get_opcode(si.raw) for si in real]
    dst = [set(x for x in _get_dest_regs(si.raw) if x < 255) for si in real]
    src = [set(x for x in _get_src_regs(si.raw) if x < 255) for si in real]
    fxu = lambda o: o in _LAYOUT or o in _IADD3
    for i in range(n - 1):
        if (fxu(opc[i]) and fxu(opc[i + 1]) and dst[i] & src[i + 1]
                and (opc[i], opc[i + 1]) not in _FORW_SAFE):
            return True                              # gap-0 non-forwarding FXU RAW
    return False


def _liveout_after(instrs, end):
    """Regs read after index `end` before being redefined (live-out of a run
    ending at `end`)."""
    seen_def = set(); live = set()
    for k in range(end, len(instrs)):
        raw = instrs[k].raw
        if _get_opcode(raw) == _NOP:
            continue
        for r in _get_src_regs(raw):
            if r < 255 and r not in seen_def:
                live.add(r)
        for r in _get_dest_regs(raw):
            if r < 255:
                seen_def.add(r)
    return live


def _sched_rename_run(run, liveout, L, reserved=frozenset()):
    real = [si for si in run if _get_opcode(si.raw) != _NOP]
    n = len(real)
    if n < 2:
        return list(run)
    raws = [si.raw for si in real]
    op = [_get_opcode(r) for r in raws]
    gwr = [set(x for x in _get_dest_regs(r) if x < 255) for r in raws]
    grd = [set(x for x in _get_src_regs(r) if x < 255) for r in raws]
    pio = [_pred_io(r) for r in raws]

    # last writer of each reg in the run
    last_wr = {}
    for i in range(n):
        for r in gwr[i]:
            last_wr[r] = i

    # ---- pass 1: SSA value assignment (renamability decided in pass 2) ----
    cur = {}                       # orig reg -> value id
    vprod = []; vreg = []; visout = []
    isrc = [[] for _ in range(n)]; idst = [None]*n
    def newval(prod, reg):
        v = len(vprod); vprod.append(prod); vreg.append(reg); visout.append(False); return v
    idsts = [dict() for _ in range(n)]      # i -> {dest reg: value} (ALL dests incl pairs)
    for i in range(n):
        for r in sorted(grd[i]):
            if r not in cur:
                cur[r] = newval(None, r)        # live-in
            isrc[i].append((r, cur[r]))
        for r in sorted(gwr[i]):
            v = newval(i, r)
            if last_wr[r] == i and r in liveout:
                visout[v] = True                # final write of a live-out reg
            cur[r] = v
            idsts[i][r] = v
            if op[i] in _LAYOUT and idst[i] is None:
                idst[i] = (r, v)                # simple-ALU single dest (for reencode)
    nv = len(vprod)
    readers = [[] for _ in range(nv)]
    for i in range(n):
        for (_r, v) in isrc[i]:
            readers[v].append(i)

    # ---- pass 2: PER-VALUE renamability ----
    # A value is renamable (gets a fresh reg, breaking false WAR/WAW) iff: it has
    # a SIMPLE single-dest producer, is NOT live-out, and EVERY reader is a
    # simple-ALU op we can re-encode.  Non-renamable values keep their orig reg,
    # so non-simple readers (and cross-run consumers) always see the right reg.
    # This is PER-VALUE (not per-reg): other values that reuse the same phys reg
    # can still be renamed.
    vphys_pin = [vreg[v] for v in range(nv)]    # default: pinned to original reg
    if _RENAME:
        for v in range(nv):
            p = vprod[v]
            if (p is not None and op[p] in _LAYOUT and len(gwr[p]) == 1
                    and not visout[v]
                    and all(op[r] in _LAYOUT for r in readers[v])):
                vphys_pin[v] = None             # renamable

    # ---- dependency DAG ----
    succ = [[] for _ in range(n)]; npred = [0]*n
    seen_edge = set()
    def add_edge(a, b, lat):
        if a == b: return
        key = (a, b)
        if key in seen_edge:                    # keep max latency for a dup edge
            for idx, (bb, ll) in enumerate(succ[a]):
                if bb == b:
                    if lat > ll: succ[a][idx] = (b, lat)
                    return
        seen_edge.add(key); succ[a].append((b, lat)); npred[b] += 1
    # RAW via GPR values
    for i in range(n):
        ps = {}
        for (_r, v) in isrc[i]:
            p = vprod[v]
            if p is not None:
                lat = 1 if (op[p], op[i]) in _FORW_SAFE else L   # forwarding-safe -> no gap
                ps[p] = max(ps.get(p, 0), lat)
        # pred RAW — carry/guard from an ALU op is an FXU result needing the
        # full FXU->FXU latency L (the store-guard IADD3->IADD3.X carry hazard).
        for pr in pio[i][1]:
            for j in range(i-1, -1, -1):
                if pr in pio[j][0]:
                    lat = L if (op[j] in _LAYOUT or op[j] in _IADD3) else 1
                    ps[j] = max(ps.get(j, 0), lat); break
        for j, lat in ps.items():
            add_edge(j, i, lat)
    # GPR WAR/WAW on PINNED physical regs (incl. ALL pair dests).  Renamed values
    # use fresh regs so they carry no false anti-dependency.
    lastdef = {}; lastuse = {}
    for i in range(n):
        wregs = {r for r in gwr[i] if vphys_pin[idsts[i][r]] is not None}
        rregs = {r for (r, v) in isrc[i] if vphys_pin[v] is not None}
        for r in wregs:                            # WAW + WAR
            if r in lastdef: add_edge(lastdef[r], i, 1)
            for u in lastuse.get(r, ()): add_edge(u, i, 1)
        for r in rregs:                            # RAW (dup-safe; main RAW above)
            if r in lastdef: add_edge(lastdef[r], i, 1)
        for r in wregs: lastdef[r] = i; lastuse[r] = []
        for r in rregs: lastuse.setdefault(r, []).append(i)
    # PREDICATE anti-deps with LIVENESS.  A pred write is DEAD if no read occurs
    # before the next write of that pred and it isn't the final (live-out) write.
    # Dead writes (independent lanes' dead carry-P0) skip WAW so the lanes don't
    # serialize; they're only kept before the NEXT LIVE write so they can't
    # clobber a live carry/guard value.  (Pred RAW is added in the section above.)
    p_writes = {}; p_reads = {}
    for i in range(n):
        for p in pio[i][0]: p_writes.setdefault(p, []).append(i)
        for p in pio[i][1]: p_reads.setdefault(p, []).append(i)
    for p, ws in p_writes.items():
        rs = p_reads.get(p, [])
        live = [i for idx, i in enumerate(ws)
                if idx == len(ws) - 1
                or any(i < r < (ws[idx + 1] if idx + 1 < len(ws) else n) for r in rs)]
        live_set = set(live)
        for a, b in zip(live, live[1:]):           # WAW among consecutive LIVE writes
            add_edge(a, b, 1)
        for r in rs:                               # WAR: read -> next write
            k = _bisect.bisect_right(ws, r)
            if k < len(ws): add_edge(r, ws[k], 1)
        for i in ws:                               # dead write -> next LIVE write
            if i not in live_set:
                k = _bisect.bisect_right(live, i)
                if k < len(live): add_edge(i, live[k], 1)

    # ---- list schedule (critical-path priority) ----
    cp = [1]*n
    for i in range(n-1, -1, -1):
        if succ[i]: cp[i] = 1 + max(lat + cp[k] for (k, lat) in succ[i])
    earliest = [0]*n; rem = npred[:]
    ready = {i for i in range(n) if rem[i] == 0}
    order = []; cyc = 0; done = 0
    while done < n:
        avail = [i for i in ready if earliest[i] <= cyc]
        if avail:
            pk = max(avail, key=lambda i: (cp[i], -i)); ready.discard(pk)
            order.append(pk); done += 1
            for (k, lat) in succ[pk]:
                earliest[k] = max(earliest[k], cyc + lat); rem[k] -= 1
                if rem[k] == 0: ready.add(k)
        else:
            order.append(None)
        cyc += 1
        if cyc > 64 * n:
            return list(run)                       # safety: bail, keep original

    # ---- physical allocation (linear scan over scheduled order) ----
    pinned_regs = set(vphys_pin[v] for v in range(nv) if vphys_pin[v] is not None)
    pool = [r for r in range(254, 1, -1)
            if r not in pinned_regs and r not in reserved]
    phys = {}
    for v in range(nv):
        if vphys_pin[v] is not None: phys[v] = vphys_pin[v]
    remuse = [len(readers[v]) for v in range(nv)]
    maxreg = [max(pinned_regs) if pinned_regs else 1]

    def reencode(raw, dst_phys, src_phys_list):
        b = bytearray(raw); db, sbs = _LAYOUT[_get_opcode(raw)]
        b[db] = dst_phys; maxreg[0] = max(maxreg[0], dst_phys)
        si = 0
        for bb in sbs:
            if raw[bb] == 255: continue
            b[bb] = src_phys_list[si]; maxreg[0] = max(maxreg[0], src_phys_list[si]); si += 1
        return bytes(b)

    # Reuse cooldown: a freed fresh reg is not reusable until L scheduled
    # positions later, so the prior occupant's FXU write-back completes before a
    # new value overwrites it.  Without this, schedule-then-allocate reuse adds a
    # WAR/WAW the scheduled DAG never modeled -> intermittent corruption.
    cooling = []                                       # [(ready_pos, reg)]
    outsi = []
    for q, slot in enumerate(order):
        # release regs whose cooldown has elapsed
        if cooling:
            keep = []
            for rp, rg in cooling:
                (pool.append(rg) if rp <= q else keep.append((rp, rg)))
            cooling = keep
        if slot is None:
            outsi.append(SassInstr(_enc_nop(), 'NOP  // sched latency')); continue
        i = slot
        # allocate dest value
        if idst[i] is not None:
            r, v = idst[i]
            if v not in phys:
                if vphys_pin[v] is not None:
                    phys[v] = r
                elif pool:
                    phys[v] = pool.pop()
                elif cooling:                          # pool dry: take soonest-ready (rare)
                    cooling.sort(); phys[v] = cooling.pop(0)[1]
                else:
                    phys[v] = r
        # build instruction.  RE-ENCODE every simple-ALU op (dest = its value's
        # phys — original if pinned, fresh if renamed — and EACH src = its
        # value's phys).  This rewrites renamed sources even when the dest is
        # pinned (live-out).  Non-simple ops only read pinned/live-in regs (the
        # nonsimple_read guard forbids renaming any reg a non-simple op reads),
        # so they are emitted unchanged.
        if op[i] in _LAYOUT and idst[i] is not None:
            _, sbs = _LAYOUT[op[i]]
            srcmap = {r: v for (r, v) in isrc[i]}
            srcphys = []
            for bb in sbs:
                rg = raws[i][bb]
                if rg == 255: continue
                srcphys.append(phys[srcmap[rg]] if rg in srcmap else rg)
            outsi.append(SassInstr(reencode(raws[i], phys[idst[i][1]], srcphys),
                                   real[i].comment))
        else:
            outsi.append(real[i])
        # free dead source values -> cooldown (reusable L positions later)
        for (r, v) in isrc[i]:
            remuse[v] -= 1
            if remuse[v] == 0 and vphys_pin[v] is None and not visout[v]:
                cooling.append((q + _COOL, phys[v]))
    if os.environ.get("OPENPTXAS_LIST_SCHED_DEBUG"):
        import sys as _sys
        nst = sum(1 for s in order if s is None)
        nrn = sum(1 for v in range(nv) if vphys_pin[v] is None)
        _sys.stderr.write(f"[ls] real={n} stalls={nst} out={len(outsi)} "
                          f"renamed={nrn}/{nv} cp={max(cp)} maxreg={maxreg[0]}\n")
    return outsi
