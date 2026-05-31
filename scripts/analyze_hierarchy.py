import json

f = 'data/manuals/new_chunk/chunk_integrate4/manual_35519890__Motherboard.jsonl'
chunks = []
with open(f, 'r', encoding='utf-8') as fh:
    for line in fh:
        chunks.append(json.loads(line))

# Basic stats
tiers = {}
for c in chunks:
    t = c.get('retrieval_tier', 'unknown')
    tiers.setdefault(t, 0)
    tiers[t] += 1

print('Total chunks:', len(chunks))
print('Tier distribution:')
for t, cnt in sorted(tiers.items()):
    print(f'  {t}: {cnt}')

# Hierarchy analysis
parent_ids = set(c.get('parent_chunk_id', '') for c in chunks if c.get('parent_chunk_id'))
chunk_map = {c['chunk_id']: c for c in chunks}
chunks_that_are_parents = set(cid for cid in parent_ids if cid in chunk_map)
print(f'Chunks that are parents: {len(chunks_that_are_parents)}')

# Depth distribution
depth_map = {}
for c in chunks:
    sp = c.get('section_path', [])
    d = len(sp)
    depth_map.setdefault(d, 0)
    depth_map[d] += 1
print('Depth distribution:')
for d in sorted(depth_map):
    print(f'  depth {d}: {depth_map[d]}')

# BIOS section analysis
print()
print('=== BIOS section hierarchy ===')
bios_chunks = [c for c in chunks if len(c.get('section_path', [])) > 0 and c['section_path'][0] == 'BIOS information']
print(f'BIOS total chunks: {len(bios_chunks)}')

for tier in ['support', 'primary', 'auxiliary']:
    tier_chunks = [c for c in bios_chunks if c.get('retrieval_tier') == tier]
    print(f'  {tier}: {len(tier_chunks)}')
    if tier == 'support':
        for c in tier_chunks[:10]:
            sp = c.get('section_path', [])
            title = str(c.get('title', ''))[:50]
            idx_len = len(str(c.get('index_text', '')))
            cid = c['chunk_id']
            pid = c.get('parent_chunk_id', '')
            print(f'    id={cid}  depth={len(sp)}  idx_len={idx_len}')
            print(f'    title={title}')
            print(f'    parent={pid}')
            print()

# Check: how many support at each depth?
print('=== Support chunks by depth ===')
support_depth = {}
for c in bios_chunks:
    if c.get('retrieval_tier') == 'support':
        d = len(c.get('section_path', []))
        support_depth.setdefault(d, 0)
        support_depth[d] += 1
for d in sorted(support_depth):
    print(f'  depth {d}: {support_depth[d]} support chunks')

# Check: are there chunks at depth 2 that are NOT support?
print()
print('=== Depth 2 chunks (should be support but might not be) ===')
for c in bios_chunks:
    sp = c.get('section_path', [])
    if len(sp) == 2:
        tier = c.get('retrieval_tier', '')
        title = str(c.get('title', ''))[:50]
        cid = c['chunk_id']
        pid = c.get('parent_chunk_id', '')
        print(f'  {cid}  tier={tier}  title={title}')
        print(f'    parent={pid}')
