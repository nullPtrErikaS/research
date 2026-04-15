import os

def find_file(names):
    for n in names:
        if os.path.exists(n):
            print(f"  ✓ Found: {n}")
            return n
        else:
            print(f"  ✗ Not found: {n}")
    return None

def bundle_candidates(filename):
    roots = ['artifacts', 'artifacts/preproc_default', 'artifacts/newsgroups', '']
    seen = set()
    candidates = []
    for r in roots:
        path = f"{r}/{filename}" if r else filename
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates

print("Testing path finding for coords_tsne.npy:")
candidates = bundle_candidates('coords_tsne.npy')
print(f"Candidates: {candidates}\n")

result = find_file(candidates)
print(f"\nResult: {result}")

# Also check direct file access
print(f"\n\nDirect check:")
print(f"  os.path.exists('artifacts/coords_tsne.npy'): {os.path.exists('artifacts/coords_tsne.npy')}")
print(f"  os.path.abspath('artifacts/coords_tsne.npy'): {os.path.abspath('artifacts/coords_tsne.npy')}")
