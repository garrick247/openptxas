"""Empirical: does running ours's merkle K times IN ONE PROCESS (same buffers,
re-zeroed each time) still expose the non-determinism?  If yes, in-process is
usable AND device pointers are stable across launches (kills malloc-noise in
the register-dump locator).  If outputs collapse to one value, the prior
'context poisoning' caveat holds and we need subprocess isolation + a
pointer-masking strategy."""
import sys, ctypes
sys.path[:0] = ["/home/garrick/forge-workbench", "/home/garrick/opencuda",
                "/home/garrick/openptxas"]
import workbench
from sass.pipeline import compile_ptx_source

KERNEL = "merkle_hash_nodes"; N = 256; SZ = 1 << 20
children = bytes((i*7+3) & 0xff for i in range(SZ))
PTX = open("/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx").read()
K = int(sys.argv[1]) if len(sys.argv) > 1 else 12

cubin = compile_ptx_source(PTX)[KERNEL]
ctx = workbench.CUDAContext(); ctx.load(cubin)
f = ctx.get_func(KERNEL)
cd = ctx.alloc(SZ); ctx.copy_to(cd, children)
pd = ctx.alloc(SZ)
print(f"parents ptr (stable within proc): 0x{pd:x}")
tally = {}
for i in range(K):
    ctx.copy_to(pd, b"\x00" * 64)                       # re-zero parent0
    vals = [ctypes.c_uint64(cd), ctypes.c_uint64(N), ctypes.c_uint64(pd),
            ctypes.c_uint64(N), ctypes.c_uint64(N)]
    a = (ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    ctx.cuda.cuLaunchKernel(f, (N+255)//256,1,1, 256,1,1, 0, None, a, None)
    ctx.sync()
    h = bytes(ctx.copy_from(pd, SZ)[0:32]).hex()
    tally[h] = tally.get(h, 0) + 1
    print(f"run {i}: {h[:48]}")
print(f"\ndistinct over {K} in-process runs: {len(tally)}")
print("VERDICT:", "DETERMINISTIC (in-proc hides race)" if len(tally)==1
      else "NON-DETERMINISTIC (in-proc exposes race)")
