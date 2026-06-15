#!/usr/bin/env python3
"""
Multi-entry differential GPU harness for FORGE kernels (openptxas vs ptxas).

Fixes the two limitations of /tmp/driver.py:
  1. Per-ENTRY parsing.  A FORGE PTX module can declare several .entry
     kernels (gather.ptx = gather_u32 + gather_u256); the old harness grabbed
     the first entry name and concatenated every entry's params.
  2. openptxas multi-entry cubins.  `__main__.py --out` emits a single cubin
     whose entry symbols cuModuleGetFunction can't find (err 500).  Instead we
     use compile_ptx_source(), which returns ONE single-entry cubin per
     kernel; ptxas keeps its multi-entry cubin and we select each symbol.

For each (module, entry) it launches the ptxas cubin and the openptxas cubin
with identical inputs (/tmp/fixin.bin) at N in {256, 4096} and compares the
output-buffer hash:  MATCH / OURS WRONG OUTPUT / OURS BROKEN / harness-skip.

Strength note: inputs are a single shared fixture (MDRIVER_FIXIN) marshalled
by name heuristics (worker.py), so they do not deeply exercise every kernel
(some share output hashes).  It is a sound openptxas-vs-ptxas DIVERGENCE
check (per kernel ptxas==ours), anchored by bit_reverse_qm31 whose output is
meaningful and size-varying.  For a stronger per-kernel oracle, supply real
index/scale buffers.

Dependencies (all on the linux 5090 box; override via env):
  MDRIVER_OPENPTXAS  openptxas repo root      (default ~/openptxas)
  MDRIVER_PTXDIR     dir of FORGE *.ptx        (default ~/forge/analysis/vortex_ntt)
  MDRIVER_PTXAS      ptxas binary              (default /usr/local/cuda/bin/ptxas)
  MDRIVER_WORKER     per-cubin launch worker   (default alongside this file)
worker.py needs forge-workbench (workbench.CUDAContext) + MDRIVER_FIXIN.

openptxas honours OPENPTXAS_ENABLE_* env vars set on THIS process (compile is
in-process).  Run with:
    export PATH=/usr/local/cuda/bin:$PATH
    OPENPTXAS_ENABLE_SHL_DISTRIBUTE=1 OPENPTXAS_ENABLE_NONHZ_LEA=1 \
        python3 mdriver.py [kernel_substr ...]
"""
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOME = os.path.expanduser('~')
OPENPTXAS = os.environ.get('MDRIVER_OPENPTXAS', os.path.join(_HOME, 'openptxas'))
sys.path.insert(0, OPENPTXAS)
from sass.pipeline import compile_ptx_source  # noqa: E402

PTXDIR = os.environ.get('MDRIVER_PTXDIR',
                        os.path.join(_HOME, 'forge/analysis/vortex_ntt'))
PTXAS = os.environ.get('MDRIVER_PTXAS', '/usr/local/cuda/bin/ptxas')
WORKER = os.environ.get('MDRIVER_WORKER', os.path.join(_HERE, 'worker.py'))
SIZES = [(256, 8), (4096, 12)]
_PARAM_RE = re.compile(r'\.param\s+\.(u64|u32|s32|f32)\s+([A-Za-z0-9_$]+)')
_ENTRY_RE = re.compile(r'\.entry\s+([A-Za-z0-9_$]+)\s*\(')


def parse_entries(text):
    """Return [(name, [(ty, pname), ...]), ...] — params scoped per entry."""
    out = []
    for m in _ENTRY_RE.finditer(text):
        name = m.group(1)
        i = m.end() - 1            # at the '('
        depth = 0
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        sig = text[m.end():i]
        out.append((name, _PARAM_RE.findall(sig)))
    return out


def worker(cubin, sym, n, lg, params):
    args = ['python3', WORKER, cubin, sym, str(n), str(lg)]
    args += [f'{ty}:{nm}' for ty, nm in params]
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return ('CRASH', '-')
    mo = re.search(r'SYNC (\S+) OUT (\S+)', r.stdout)
    return (mo.group(1), mo.group(2)) if mo else ('NOOUT', '-')


def main():
    filt = sys.argv[1:]
    print(f"{'module/entry':40s} {'N':>5s}  ptxas        ours         verdict")
    for fn in sorted(os.listdir(PTXDIR)):
        if not fn.endswith('.ptx'):
            continue
        if filt and not any(f in fn for f in filt):
            continue
        path = os.path.join(PTXDIR, fn)
        text = open(path).read()
        entries = parse_entries(text)
        # ptxas: one multi-entry cubin for the whole module.
        rc = f'/tmp/m_{fn}.r.cubin'
        if subprocess.run([PTXAS, '-arch=sm_120', path, '-o', rc],
                          capture_output=True).returncode:
            print(f'{fn:40s} PTXAS-FAIL')
            continue
        # openptxas: per-kernel single-entry cubins.
        try:
            ours = compile_ptx_source(text)
        except Exception as e:                                   # noqa: BLE001
            print(f'{fn:40s} OURS-COMPILE-FAIL {repr(e)[:50]}')
            continue
        for name, params in entries:
            if name not in ours:
                print(f'{fn}/{name:30s} OURS-MISSING-ENTRY')
                continue
            oc = f'/tmp/m_{fn}.{name}.o.cubin'
            open(oc, 'wb').write(ours[name])
            for n, lg in SIZES:
                rs, rh = worker(rc, name, n, lg, params)
                os_, oh = worker(oc, name, n, lg, params)
                if rs != '0':
                    v = 'skip (ptxas not clean)'
                elif os_ != '0':
                    v = '*** OURS BROKEN ***'
                elif rh == oh:
                    v = 'MATCH'
                else:
                    v = '*** OURS WRONG OUTPUT ***'
                print(f'{fn[:-4]}/{name:28s} {n:>5d}  '
                      f's={rs:>4s} {rh:9s}  s={os_:>4s} {oh:9s}  {v}')


if __name__ == '__main__':
    main()
