#!/usr/bin/env python3
"""
Per-cubin GPU launch worker for the differential harness (mdriver.py).

Loads ONE cubin, looks up ONE entry symbol, marshals args from the entry's
PTX param list (name heuristics), launches on the GPU, and prints
    SYNC <code> OUT <md5[:12] of the output buffers>
so the parent (mdriver.py) can compare openptxas vs ptxas.  Kept in its own
process so a bad cubin's illegal-address fault is isolated to one launch.

Usage:  worker.py <cubin> <sym> <N> <LOG_N> <ty:pname> ...

Env (all default to the linux 5090 box layout):
  MDRIVER_WORKBENCH  dir with workbench.CUDAContext (default ~/forge-workbench)
  MDRIVER_OPENCUDA   opencuda repo root              (default ~/opencuda)
  MDRIVER_OPENPTXAS  openptxas repo root             (default ~/openptxas)
  MDRIVER_FIXIN      input fixture blob              (default /tmp/fixin.bin)
"""
import ctypes
import hashlib
import os
import sys

_HOME = os.path.expanduser('~')
sys.path[:0] = [
    os.environ.get('MDRIVER_WORKBENCH', os.path.join(_HOME, 'forge-workbench')),
    os.environ.get('MDRIVER_OPENCUDA', os.path.join(_HOME, 'opencuda')),
    os.environ.get('MDRIVER_OPENPTXAS', os.path.join(_HOME, 'openptxas')),
]
import workbench  # noqa: E402

cubin, sym, N, LG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
params = [p.split(':') for p in sys.argv[5:]]    # "ty:name"
BUF = 1 << 20
IN = open(os.environ.get('MDRIVER_FIXIN', '/tmp/fixin.bin'), 'rb').read()


def _args(*vals):
    arr = (ctypes.c_void_p * len(vals))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    return arr, vals


ctx = workbench.CUDAContext()
if not ctx.load(open(cubin, 'rb').read()):
    print('SYNC LOADFAIL OUT -')
    sys.exit(0)
f = ctx.get_func(sym)

bufs = {}
vals = []
for ty, name in params:
    ln = name.lower()
    if ty == 'u64' and ('data' in ln or 'ptr' in ln or ln.endswith('buf')):
        is_out = ('out' in ln or 'dst' in ln)
        d = ctx.alloc(BUF)
        ctx.copy_to(d, b'\x00' * BUF if is_out else IN)
        bufs[name] = (d, is_out)
        vals.append(ctypes.c_uint64(d))
    elif 'log' in ln:
        vals.append(ctypes.c_uint32(LG))
    elif ty == 'u64':
        vals.append(ctypes.c_uint64(N))
    else:
        vals.append(ctypes.c_uint32(N))

args, _hold = _args(*vals)
ctx.cuda.cuLaunchKernel(f, (N + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None)
s = ctx.sync()

out_names = [n for n, (d, o) in bufs.items() if o] or list(bufs)
h = hashlib.md5(b''.join(ctx.copy_from(bufs[n][0], BUF)
                        for n in sorted(out_names))).hexdigest()[:12]
print(f'SYNC {s} OUT {h}')
