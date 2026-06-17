"""Compare ctrl-word fields of ARX ALU ops between ours and ptxas merkle cubins,
parsed from `nvdisasm -hex`.  ctrl = (high64 >> 41) & 0x7fffff.
Layout: [22:17]stall [16]yield [15]wbar [14:10]rbar [9:4]wdep [3:0]misc."""
import sys, re, subprocess, tempfile
from collections import Counter
sys.path.insert(0, "/home/garrick/openptxas")
from sass.pipeline import compile_ptx_source

PTX = open("/home/garrick/forge/analysis/vortex_ntt/merkle_hash_nodes.ptx").read()
KERNEL = "merkle_hash_nodes"
ALU_MNEM = ("LOP3", "SHF.L", "SHF.R", "IADD3", "IADD ")

def decode(cubin):
    out = subprocess.run(["/usr/local/cuda/bin/nvdisasm", "-c", "-hex", cubin],
                         capture_output=True, text=True).stdout.splitlines()
    rows, i = [], 0
    while i < len(out):
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?)\s*/\* 0x[0-9a-f]+ \*/\s*$", out[i])
        hw = re.match(r"\s*/\* (0x[0-9a-f]+) \*/\s*$", out[i+1]) if i+1 < len(out) else None
        if m and hw:
            text = m.group(1).rstrip(" ;")
            text = re.sub(r"^@!?U?P\S*\s+", "", text)        # strip predicate
            rows.append((text, (int(hw.group(1), 16) >> 41) & 0x7FFFFF))
            i += 2
        else:
            i += 1
    return rows

def survey(cubin, label):
    rows = decode(cubin)
    print(f"\n=== {label} ({len(rows)} instrs) ===")
    by = {}
    for mnem, c in rows:
        key = next((a.strip() for a in ALU_MNEM if mnem.startswith(a)), None)
        if not key: continue
        by.setdefault(key, []).append(dict(
            stall=(c >> 17) & 0x3f, yld=(c >> 16) & 1, wbar=(c >> 15) & 1,
            rbar=(c >> 10) & 0x1f, wdep=(c >> 4) & 0x3f))
    for name, cs in sorted(by.items()):
        print(f"  {name:7s} n={len(cs):4d}  "
              f"stall={dict(Counter(c['stall'] for c in cs))}  "
              f"rbar={dict(Counter(hex(c['rbar']) for c in cs))}  "
              f"wdep={dict(Counter(hex(c['wdep']) for c in cs))}")

oc = compile_ptx_source(PTX)[KERNEL]; open("/tmp/cmp_o.cubin","wb").write(oc)
tf = tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False); tf.write(PTX); tf.close()
subprocess.run(["/usr/local/cuda/bin/ptxas","-arch=sm_120",tf.name,"-o","/tmp/cmp_p.cubin"],check=True)
survey("/tmp/cmp_o.cubin", "OURS")
survey("/tmp/cmp_p.cubin", "PTXAS")
