"""Compare ours-emu (ideal, deterministic) to ptxas-on-GPU (correct) with the
SAME input. MATCH => ours's logic is now correct, residual GPU non-determinism
is a hazard. DIFFER => another logic bug in ours; the emulator's per-instruction
trace can localize it."""
import sys, ctypes, struct, subprocess, tempfile
sys.path[:0] = ["/home/garrick/forge-workbench", "/home/garrick/opencuda", "/home/garrick/openptxas"]
import workbench
import tools.sass_emu as E
from sass.pipeline import compile_ptx_source

PTXPATH = "/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx"
PTX = open(PTXPATH).read()
KERNEL = "merkle_hash_nodes"
N = 256
SZ = 1 << 20
CHILD_BASE = 0x10000000
PAR_BASE = 0x20000000
children = bytes((i * 7 + 3) & 0xff for i in range(SZ))

# ---- GPU run (ptxas reference) ----
def gpu_run(cubin_bytes):
    ctx = workbench.CUDAContext()
    if not ctx.load(cubin_bytes):
        print("GPU LOAD FAILED"); return None
    f = ctx.get_func(KERNEL)
    cd = ctx.alloc(SZ); ctx.copy_to(cd, children)
    pd = ctx.alloc(SZ); ctx.copy_to(pd, b"\x00" * SZ)
    vals = [ctypes.c_uint64(cd), ctypes.c_uint64(N), ctypes.c_uint64(pd),
            ctypes.c_uint64(N), ctypes.c_uint64(N)]
    a = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    le = ctx.cuda.cuLaunchKernel(f, (N + 255) // 256, 1, 1, 256, 1, 1, 0, None, a, None)
    se = ctx.sync()
    if le or se:
        print(f"GPU launch={le} sync={se}"); return None
    return bytes(ctx.copy_from(pd, SZ)[0:32])

# ---- Emulator run (ours, ideal) ----
def emu_run(cubin_path):
    s = E.new_state(); s.ntid = (256, 1, 1); s.ctaid = (0, 0, 0); s.tid = (0, 0, 0)
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
    ins = E.decode_kernel(cubin_path, KERNEL)
    E.run(s, ins)
    return bytes(s.globals[PAR_BASE][0:32])

out = compile_ptx_source(PTX); open("/tmp/g_o.cubin", "wb").write(out[KERNEL])
tf = tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False); tf.write(PTX); tf.close()
subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_120", tf.name, "-o", "/tmp/g_p.cubin"], check=True)

gpu = gpu_run(open("/tmp/g_p.cubin", "rb").read())
emu = emu_run("/tmp/g_o.cubin")
print("ptxas-GPU parent0:", gpu.hex() if gpu else None)
print("ours-emu  parent0:", emu.hex())
if gpu:
    print("MATCH => ours logic correct; residual GPU non-det is a HAZARD" if gpu == emu
          else "DIFFER => another LOGIC bug in ours (emulator can localize it)")
