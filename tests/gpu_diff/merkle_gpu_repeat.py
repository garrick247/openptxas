"""Compile ours's merkle cubin once, then launch it K times in fresh
subprocesses (reliable isolation) and tally distinct parent0 values.

Determinism verdict for ours's GPU output, comparable to the emulator's
5377e4ff... ground truth.

Usage:  merkle_gpu_repeat.py [K]   (default K=8)
"""
import sys, os, subprocess
sys.path.insert(0, "/home/garrick/openptxas")
from sass.pipeline import compile_ptx_source

HERE = os.path.dirname(os.path.abspath(__file__))
PTX = open("/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx").read()
KERNEL = "merkle_hash_nodes"
EMU_GT = "5377e4ff957bda4d4535f4879876b71a61056c4cec31e78397c66ec47a86a130"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8

cubin = compile_ptx_source(PTX)[KERNEL]
open("/tmp/ours_merkle.cubin", "wb").write(cubin)

tally = {}
for i in range(K):
    r = subprocess.run(["python3", os.path.join(HERE, "merkle_gpu_launch1.py"),
                        "/tmp/ours_merkle.cubin"], capture_output=True, text=True, timeout=60)
    out = r.stdout.strip() or f"ERR:{r.returncode}:{r.stderr.strip()[:40]}"
    tally[out] = tally.get(out, 0) + 1
    print(f"run {i}: {out[:64]}")

print(f"\ndistinct outputs over {K} runs: {len(tally)}")
for v, c in sorted(tally.items(), key=lambda kv: -kv[1]):
    tag = "  (== emu ground truth)" if v == EMU_GT else ""
    print(f"  {c:2d}x  {v[:64]}{tag}")
print("VERDICT:", "DETERMINISTIC" if len(tally) == 1 else "NON-DETERMINISTIC")
