import re, sys

# Parse nvdisasm -c output into ordered (label|insn) items.
items = []
for l in open(sys.argv[1]):
    s = l.strip()
    m = re.match(r'/\*[0-9a-f]+\*/\s*(.*;)', s)
    lab = re.match(r'(\.L_x_\d+):', s)
    if lab: items.append(('label', lab.group(1)))
    elif m: items.append(('insn', m.group(1)))

# Split into blocks keyed by label (None for the entry block).
blocks = {}      # label -> list[insn]
order = []
cur_label, cur = None, []
def flush():
    blocks[cur_label] = cur; order.append(cur_label)
for k, t in items:
    if k == 'label':
        flush(); cur_label, cur = t, []
    else:
        cur.append(t)
flush()

# Valid storing-thread path: entry block -> .L_x_0 (compute) -> .L_x_1 (store).
# (Valid threads take @!P0 BRA .L_x_0, then fall through to .L_x_1; they skip
# the .L_x_1-direct path that invalid threads take.)
seq = blocks.get(None, []) + blocks.get('.L_x_0', []) + blocks.get('.L_x_1', [])
boundary_blk3 = len(blocks.get(None, [])) + len(blocks.get('.L_x_0', []))

WIDE = ('.WIDE', '.64')
def is_wide(t): return any(w in t for w in WIDE) or t.split()[0].startswith('LEA')

definite = set(['RZ'])
read_uninit = {}     # reg -> (index, text)
for idx, t in enumerate(seq):
    pred = bool(re.match(r'@!?P[T0-9]+ ', t))
    body = re.sub(r'^@!?P[T0-9]+ ', '', t)
    op = body.split()[0]
    regs = re.findall(r'\bR[0-9]+\b', body)
    if not regs:
        continue
    is_store = any(x in op for x in ('STG', 'STS', 'STL', 'RED', 'ATOM'))
    dst = None if is_store else regs[0]
    srcs = regs if is_store else regs[1:]
    # reads
    for r in srcs:
        if r != 'RZ' and r not in definite and r not in read_uninit:
            read_uninit[r] = (idx, t, idx >= boundary_blk3)
    # writes: only UNCONDITIONAL writes guarantee a value
    if dst and dst != 'RZ' and not pred:
        definite.add(dst)
        if is_wide(t):
            definite.add('R' + str(int(dst[1:]) + 1))

print(f"valid-path instructions: {len(seq)}  (blk3/.L_x_1 starts at idx {boundary_blk3})")
print(f"\nregisters READ before any GUARANTEED (unconditional) write, on the storing path:")
for r in sorted(read_uninit, key=lambda x: int(x[1:])):
    idx, t, in_blk3 = read_uninit[r]
    where = 'blk3/STORE' if in_blk3 else 'blk0/blk2'
    print(f"  {r}: idx {idx} ({where}): {t}")
