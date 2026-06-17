"""Single isolated GPU launch of a merkle cubin -> prints parent0 hex.
One launch per process so a fault/race is isolated (the only RELIABLE mode;
in-process repeats are context-poisoned -- see journal 2026-06-11).

Usage:  merkle_gpu_launch1.py <cubin>
Uses the SAME deterministic input as gpu_vs_emu.py (children[i]=(i*7+3)&0xff,
N=256) so a parent0 read here is directly comparable to the emulator value
5377e4ff... and across runs.
"""
import sys, ctypes
sys.path[:0] = ["/home/garrick/forge-workbench", "/home/garrick/opencuda",
                "/home/garrick/openptxas"]
import workbench

KERNEL = "merkle_hash_nodes"
N = 256
SZ = 1 << 20
children = bytes((i * 7 + 3) & 0xff for i in range(SZ))

cubin = open(sys.argv[1], "rb").read()
ctx = workbench.CUDAContext()
if not ctx.load(cubin):
    print("LOADFAIL"); sys.exit(0)
f = ctx.get_func(KERNEL)
cd = ctx.alloc(SZ); ctx.copy_to(cd, children)
pd = ctx.alloc(SZ); ctx.copy_to(pd, b"\x00" * SZ)
vals = [ctypes.c_uint64(cd), ctypes.c_uint64(N), ctypes.c_uint64(pd),
        ctypes.c_uint64(N), ctypes.c_uint64(N)]
a = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
le = ctx.cuda.cuLaunchKernel(f, (N + 255) // 256, 1, 1, 256, 1, 1, 0, None, a, None)
se = ctx.sync()
if le or se:
    print(f"LAUNCHFAIL le={le} se={se}"); sys.exit(0)
print(bytes(ctx.copy_from(pd, SZ)[0:32]).hex())
