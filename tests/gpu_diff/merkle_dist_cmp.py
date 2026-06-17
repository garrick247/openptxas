"""Measure real (non-NOP) instruction distance between RAW-dependent ALU pairs
in ours vs ptxas merkle.  Hypothesis: ours places dependent ALU ops at 0 real
instructions apart (separated only by squashed NOPs + slot-tracking), while
ptxas interleaves independent real instructions to cover ALU pipeline latency.
If ours has many 0/1-real-gap RAW pairs and ptxas has ~none, the race is a
SCHEDULER issue (instruction interleaving), not ctrl bits."""
import sys, re, subprocess, tempfile
from collections import Counter
sys.path.insert(0, "/home/garrick/openptxas")
from sass.pipeline import compile_ptx_source

PTX = open("/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx").read()
KERNEL = "merkle_hash_nodes"
ALU = ("LOP3", "SHF.L", "SHF.R", "IADD3", "IADD ", "IMAD")

def parse(cubin):
    txt = subprocess.run(["/usr/local/cuda/bin/nvdisasm","-c",cubin],
                         capture_output=True, text=True).stdout.splitlines()
    out = []
    for ln in txt:
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;", ln)
        if not m: continue
        t = re.sub(r"^@!?U?P\S*\s+", "", m.group(1)).strip()
        regs = re.findall(r"\bR(\d+)\b", t)
        regs = [int(x) for x in regs]
        is_alu = any(t.startswith(a) for a in ALU)
        is_nop = t.startswith("NOP")
        # dest = first GPR (after stripping P0/PT which aren't R); srcs = rest
        dst = regs[0] if regs else None
        srcs = set(regs[1:])
        out.append(dict(t=t, alu=is_alu, nop=is_nop, dst=dst, srcs=srcs))
    return out

def analyze(cubin, label):
    ins = parse(cubin)
    gaps = []                       # real-instruction gap for each RAW ALU->reader pair
    for i, p in enumerate(ins):
        if not p["alu"] or p["dst"] is None: continue
        real = 0
        for j in range(i+1, min(i+12, len(ins))):
            q = ins[j]
            if p["dst"] in q["srcs"]:          # first reader of the produced reg
                gaps.append(real); break
            if q["dst"] == p["dst"]:           # overwritten before any read
                break
            if not q["nop"]:
                real += 1
    c = Counter(gaps)
    tight = sum(v for g, v in c.items() if g <= 1)
    print(f"{label:6s}: RAW ALU->reader pairs={len(gaps)}  real-gap dist={dict(sorted(c.items()))}")
    print(f"         {tight} pairs at <=1 real instr ({100*tight//max(len(gaps),1)}%)")

oc = compile_ptx_source(PTX)[KERNEL]; open("/tmp/d_o.cubin","wb").write(oc)
tf = tempfile.NamedTemporaryFile("w",suffix=".ptx",delete=False); tf.write(PTX); tf.close()
subprocess.run(["/usr/local/cuda/bin/ptxas","-arch=sm_120",tf.name,"-o","/tmp/d_p.cubin"],check=True)
analyze("/tmp/d_o.cubin","OURS")
analyze("/tmp/d_p.cubin","PTXAS")
