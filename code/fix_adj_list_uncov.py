#!/usr/bin/env python3
"""One-time migration: rewrite adj-list files from (edge, prev_uncov) to
(edge, uncov_after_edge).

Older distributed_robust_prune.py runs recorded, for each edge, the uncov count
from *before* that edge took effect (a one-round recording lag). This script
shifts each source's uncov values by one so that tuple i stores the uncov after
edge i takes effect:

    new_uncov[i] = old_uncov[i + 1]   for i < last
    new_uncov[last] = 0               (the last edge drove uncov to 0, which is
                                       why pruning stopped there)

The edge ids (neighbor ids) are left untouched; only the uncov field moves.

Line formats handled (matching edge_to_coverage_analysis.py):
  * bare neighborhood:      [(n0, u0), (n1, u1), ...]
  * source-prefixed:        "<source_id> [(n0, u0), ...]"  (set-cover / legacy)

Usage:
    # rewrite in place, keeping a .bak copy
    python fix_adj_list_uncov.py path/to/adj-list-*.txt

    # write to a new file instead of in place
    python fix_adj_list_uncov.py --out-suffix .fixed path/to/adj-list-foo.txt

    # preview counts without writing
    python fix_adj_list_uncov.py --dry-run path/to/adj-list-*.txt
"""
import argparse
import ast
import glob
import os
import sys


def shift_neighborhood(neighborhood):
    """[(n_i, prev_uncov_i)] -> [(n_i, uncov_after_edge_i)] with last -> 0."""
    deg = len(neighborhood)
    out = []
    for i, (neighbor, _prev_uncov) in enumerate(neighborhood):
        new_uncov = neighborhood[i + 1][1] if i + 1 < deg else 0
        out.append((neighbor, new_uncov))
    return out


def parse_line(line):
    """Return (prefix, neighborhood) where prefix is '' or '<source_id> '."""
    line = line.rstrip("\n")
    if not line.strip():
        return None
    if line.lstrip().startswith("["):
        return "", ast.literal_eval(line.strip())
    # source-prefixed: "<id> [ ... ]"
    space = line.index(" ")
    prefix = line[: space + 1]
    return prefix, ast.literal_eval(line[space + 1 :])


def format_line(prefix, neighborhood):
    return f"{prefix}{neighborhood}\n"


def process_file(path, out_path, dry_run):
    n_sources = 0
    n_empty = 0
    out_lines = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            parsed = parse_line(line)
            if parsed is None:
                out_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            prefix, neighborhood = parsed
            if len(neighborhood) == 0:
                n_empty += 1
            shifted = shift_neighborhood(neighborhood)
            out_lines.append(format_line(prefix, shifted))
            n_sources += 1

    print(f"  {path}: {n_sources} sources ({n_empty} empty)")
    if dry_run:
        return

    if out_path == path:
        bak = path + ".bak"
        if not os.path.exists(bak):
            os.rename(path, bak)
        else:
            print(f"  WARNING: {bak} already exists; not overwriting backup", file=sys.stderr)
    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"  -> wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="adj-list files (globs allowed)")
    p.add_argument("--out-suffix", default=None,
                   help="write to <path><suffix> instead of in place. When "
                        "omitted, files are rewritten in place with a .bak copy.")
    p.add_argument("--dry-run", action="store_true",
                   help="parse and report counts without writing anything")
    args = p.parse_args()

    files = []
    for pat in args.paths:
        matched = glob.glob(pat)
        if not matched:
            print(f"WARNING: no files match {pat!r}", file=sys.stderr)
        files.extend(matched)
    if not files:
        p.error("no input files")

    print(f"Processing {len(files)} file(s){' (dry run)' if args.dry_run else ''}")
    for path in sorted(files):
        out_path = path + args.out_suffix if args.out_suffix else path
        process_file(path, out_path, args.dry_run)


if __name__ == "__main__":
    main()
