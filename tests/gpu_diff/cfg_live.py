import re, sys

# Read SASS with labels/branches preserved.
lines = [l.rstrip() for l in open(sys.argv[1])]
# Keep instruction text + labels.
insns = []   # (kind, text)  kind in {'label','insn'}
for l in lines:
    s = l.strip()
    m = re.match(r'/\*[0-9a-f]+\*/\s*(.*;)', s)
    lab = re.match(r'(\.L_x_\d+):', s)
    if lab:
        insns.append(('label', lab.group(1)))
    elif m:
        insns.append(('insn', m.group(1)))

# Build basic blocks: a new block starts at a label or after a branch.
blocks = []          # list of (label_or_None, [insn_idx...])
labelpos = {}
cur = []
cur_label = None
def flush():
    global cur, cur_label
    if cur or cur_label is not None:
        blocks.append((cur_label, cur))
    cur = []; cur_label = None
for kind, txt in insns:
    if kind == 'label':
        flush(); cur_label = txt
    else:
        cur.append(txt)
        op = txt.split()[0] if txt.split() else ''
        op = re.sub(r'^@!?P[T0-9]+', '', op).strip() or (txt.split()[1] if len(txt.split())>1 else '')
        if 'BRA' in txt or 'EXIT' in txt or op.startswith('RET'):
            flush()
flush()
for i,(lab,_) in enumerate(blocks):
    if lab: labelpos[lab] = i

def regs(t):
    t = re.sub(r'^@!?P[T0-9]+ ', '', t)
    return re.findall(r'\bR[0-9]+\b', t)

# def/use per block (use = read before any write in the block)
defs=[]; uses=[]; succ=[]
for bi,(lab,body) in enumerate(blocks):
    d=set(); u=set()
    for t in body:
        op=t.split()[0]; op=re.sub(r'^@!?P[T0-9]+','',op)
        rs=regs(t)
        if not rs:
            continue
        is_store = ('STG' in t or 'STS' in t or 'STL' in t or 'RED' in t or 'ATOM' in t)
        dst = None if is_store else rs[0]
        src = rs if is_store else rs[1:]
        # WIDE / 64-bit ops also write dst+1
        for r in src:
            if r!='RZ' and r not in d: u.add(r)
        if dst and dst!='RZ':
            d.add(dst)
            if '.WIDE' in t or '.64' in t or t.startswith('LEA') or '@!P' in t or '@P' in t:
                d.add('R'+str(int(dst[1:])+1))
    defs.append(d); uses.append(u)
    # successors: fallthrough + branch target
    s=set()
    last = body[-1] if body else ''
    cond = last.startswith('@')
    bt = re.search(r'`\((\.L_x_\d+)\)', last)
    has_uncond = ('BRA' in last and not cond) or 'BRA.U' in last
    if bt and bt.group(1) in labelpos: s.add(labelpos[bt.group(1)])
    if not has_uncond and 'EXIT' not in last and 'RET' not in last and bi+1<len(blocks):
        s.add(bi+1)
    succ.append(s)

# Liveness fixpoint
live_in=[set() for _ in blocks]; live_out=[set() for _ in blocks]
for _ in range(200):
    changed=False
    for bi in reversed(range(len(blocks))):
        lo=set()
        for sb in succ[bi]: lo|=live_in[sb]
        li=uses[bi] | (lo - defs[bi])
        if li!=live_in[bi] or lo!=live_out[bi]:
            live_in[bi]=li; live_out[bi]=lo; changed=True
    if not changed: break

# Registers live-in at the ENTRY block = used before written on some path = UNINITIALIZED.
print("blocks:", len(blocks))
print("LIVE-IN AT ENTRY (uninitialized regs read before write on some path):", sorted(live_in[0], key=lambda x:int(x[1:])))
# Show, for each, a block where it is in use[] (first read site)
for r in sorted(live_in[0], key=lambda x:int(x[1:])):
    for bi,(lab,body) in enumerate(blocks):
        if r in uses[bi]:
            hit=[t for t in body if r in regs(t)]
            print(f"  {r}: first used in block {bi} (label {lab}): {hit[0] if hit else '?'}")
            break
