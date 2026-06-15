#!/usr/bin/env python3
"""
Subprocess-isolated pass bisection for a non-deterministic openptxas miscompile.

Every step runs in a FRESH process so a poisoned CUDA context (sync-700 from a
bad cubin) can never leak into the next measurement — the flaw that made the
earlier in-process bisection read CRASH==CRASH as "deterministic".

  compile_ours(disabled) : child proc, OPENPTXAS_DISABLE_PASSES=<disabled>,
                           compile_ptx_source -> cubin file
  measure(cubin)         : run worker.py in TWO fresh procs; returns
                           (both_ok, deterministic, hash)
  GOOD(disabled)         : compiles, both runs sync==0, the two hashes agree,
                           AND the hash equals ptxas's reference

Then:
  1. baseline (nothing disabled)  -> expect BAD (reproduces the bug)
  2. all PTX passes disabled       -> GOOD? if not, the bug is SASS-level
                                      (isel/regalloc/scoreboard), not a pass
  3. greedy re-enable: minimal set of passes that must stay disabled = culprits

Run on the linux 5090 box:
    export PATH=/usr/local/cuda/bin:$PATH
    python3 tests/gpu_diff/pass_bisect.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
OPENPTXAS = os.path.join(HOME, "openptxas")
PTX = os.path.join(HOME, "forge/analysis/vortex_ntt/merkle_hash_nodes.ptx")
KERNEL = "merkle_hash_nodes"
WORKER = os.path.join(HERE, "worker.py")
PTXAS = "/usr/local/cuda/bin/ptxas"
N, LG = 256, 8

PASSES = ["unroll", "cvta_eliminate", "imm_propagate", "mul_distribute",
          "shl_distribute", "load_cse", "cvt_roundtrip_fold", "add_forward_chain",
          "bitop_imm_chain_fold", "mul_imm_chain_fold", "common_mul_sum",
          "cvt_shl_cse", "trivial_fold", "imm_add_fold", "imm_xor_fold",
          "repeated_add_reduce", "dead_self_update_dce", "copy_prop",
          "dead_mov_dce", "m31_mod_fast_path"]


def params():
    text = open(PTX).read()
    m = re.search(r"\.entry\s+" + KERNEL + r"\s*\(", text)
    i = m.end() - 1; depth = 0
    while i < len(text):
        if text[i] == "(": depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0: break
        i += 1
    return re.findall(r"\.param\s+\.(u64|u32|s32|f32)\s+([A-Za-z0-9_$]+)", text[m.end():i])


PARAMS = params()
WARGS = [f"{ty}:{nm}" for ty, nm in PARAMS]


def compile_ours(disabled, out):
    env = dict(os.environ, OPENPTXAS_DISABLE_PASSES=",".join(sorted(disabled)))
    code = (f"import sys; sys.path.insert(0,{OPENPTXAS!r});"
            f"from sass.pipeline import compile_ptx_source;"
            f"o=compile_ptx_source(open({PTX!r}).read());"
            f"open({out!r},'wb').write(o[{KERNEL!r}])")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def launch(cubin):
    r = subprocess.run([sys.executable, WORKER, cubin, KERNEL, str(N), str(LG)] + WARGS,
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return ("CRASH", None)
    mo = re.search(r"SYNC (\S+) OUT (\S+)", r.stdout)
    if not mo:
        return ("NOOUT", None)
    return (mo.group(1), mo.group(2))


def measure(cubin):
    s1, h1 = launch(cubin)
    s2, h2 = launch(cubin)
    both_ok = (s1 == "0" and s2 == "0")
    det = both_ok and h1 == h2
    return both_ok, det, (h1 if det else None)


def ptxas_ref():
    rc = "/tmp/bisect_ref.cubin"
    subprocess.run([PTXAS, "-arch=sm_120", PTX, "-o", rc], check=True)
    ok, det, h = measure(rc)
    return h if (ok and det) else None


def GOOD(disabled, ref):
    oc = "/tmp/bisect_ours.cubin"
    if not compile_ours(disabled, oc):
        return False, "COMPILEFAIL"
    ok, det, h = measure(oc)
    if not ok: return False, "CRASH/sync!=0"
    if not det: return False, "NON-DET"
    if h != ref: return False, "WRONG (det but != ptxas)"
    return True, "GOOD"


def main():
    ref = ptxas_ref()
    print(f"ptxas reference hash: {ref}  (deterministic={ref is not None})")
    if ref is None:
        print("ptxas itself non-deterministic under the harness — abort."); return

    g, why = GOOD(set(), ref)
    print(f"baseline (nothing disabled): {'GOOD' if g else 'BAD'} [{why}]")
    g_all, why_all = GOOD(set(PASSES), ref)
    print(f"all PTX passes disabled:     {'GOOD' if g_all else 'BAD'} [{why_all}]")
    if not g_all:
        print("\n=> all-PTX-passes-off is still BAD: the bug is NOT a PTX pass")
        print("   (it is in isel / regalloc / scoreboard).  Pass-bisection cannot")
        print("   localize it; needs SASS-level differential.  Stopping.")
        return

    # Greedy: re-enable each pass; if it stays GOOD, that pass isn't needed
    # disabled.  The passes that must remain disabled are the culprits.
    disabled = set(PASSES)
    for p in PASSES:
        trial = disabled - {p}
        g, why = GOOD(trial, ref)
        if g:
            disabled = trial
            print(f"  re-enabled {p:22s} -> still GOOD")
        else:
            print(f"  re-enabled {p:22s} -> BAD [{why}]  (CULPRIT, keep disabled)")
    print(f"\nMINIMAL culprit set (passes whose presence breaks merkle): {sorted(disabled)}")


if __name__ == "__main__":
    main()
