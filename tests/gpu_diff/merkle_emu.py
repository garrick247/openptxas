"""Emulate merkle (ours vs ptxas) with perfect register semantics (no timing
hazards) and compare.  MATCH => ours's instruction stream is logically correct
=> the GPU non-determinism is a hardware HAZARD.  DIFFER => a logic miscompile."""
import sys, struct, subprocess, tempfile
sys.path.insert(0, "/home/garrick/openptxas")
import tools.sass_emu as E
from sass.pipeline import compile_ptx_source

PTXPATH = "/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx"
PTX = open(PTXPATH).read()
KERNEL = "merkle_hash_nodes"
CHILD_BASE = 0x10000000
PAR_BASE = 0x20000000
SZ = 1 << 20
children = bytes((i * 7 + 3) & 0xff for i in range(SZ))


def setup(gid):
    s = E.new_state()
    s.ntid = (256, 1, 1); s.ctaid = (0, 0, 0); s.tid = (gid, 0, 0)
    cb = bytearray(0x400)
    struct.pack_into("<I", cb, 0x360, 256)         # ntid.x (gid = ctaid*ntid + tid)
    struct.pack_into("<Q", cb, 0x380, CHILD_BASE)  # children_data
    struct.pack_into("<Q", cb, 0x388, SZ)          # children_len
    struct.pack_into("<Q", cb, 0x390, PAR_BASE)    # parents_data
    struct.pack_into("<Q", cb, 0x398, 64)          # parents_len / guard bound
    struct.pack_into("<Q", cb, 0x3a0, 64)          # n_parents
    s.cbank = bytes(cb)
    s.globals[CHILD_BASE] = bytearray(children)
    s.globals[PAR_BASE] = bytearray(SZ)
    return s


def emulate(cubin, gid=0):
    instrs = E.decode_kernel(cubin, KERNEL)
    s = setup(gid)
    E.run(s, instrs)
    # summarise notable events (OOB/unimpl/decode fails)
    ev = {}
    for e in getattr(s, "events", []):
        ev[e[0]] = ev.get(e[0], 0) + 1
    return bytes(s.globals[PAR_BASE][0:64]), ev, len(instrs)


out = compile_ptx_source(PTX)
open("/tmp/e_o.cubin", "wb").write(out[KERNEL])
tf = tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False); tf.write(PTX); tf.close()
subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_120", tf.name, "-o", "/tmp/e_p.cubin"], check=True)

po, eo, no = emulate("/tmp/e_o.cubin")
pp, ep, np_ = emulate("/tmp/e_p.cubin")
print(f"ours: {no} instrs, events={eo}")
print(f"ptx : {np_} instrs, events={ep}")
print(f"ours-emu parent0[0:64]: {po.hex()}")
print(f"ptx -emu parent0[0:64]: {pp.hex()}")
if po == pp:
    print("\n==> EMULATED OUTPUTS MATCH: ours's SASS is logically equivalent to ptxas.")
    print("    => the GPU non-determinism is a HARDWARE HAZARD (timing/forwarding),")
    print("       not a logic miscompile.")
else:
    print("\n==> EMULATED OUTPUTS DIFFER: ours has a LOGIC bug in the instruction stream")
    print("    (deterministic miscompile, not just a hazard).  Find first divergent STG.")
