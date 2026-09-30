#!/usr/bin/env python3
"""Verify frozen manifests and optionally reconstructed or upstream image files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from manifest import CLASS_NAMES, DEFAULT_MANIFEST_DIR, load_manifest, load_metadata, sha256_file


def verify(manifest_dir=DEFAULT_MANIFEST_DIR, dataset_root=None, images_root=None,
           check_hashes=False) -> dict:
    manifest_dir = Path(manifest_dir)
    metadata = load_metadata(manifest_dir)
    rows = {split: load_manifest(split, manifest_dir) for split in ('train', 'test')}
    stats = {}
    id_sets = {}
    context_sets = {}
    for split, samples in rows.items():
        counts = Counter(r['class_name'] for r in samples)
        id_counts = Counter(r['image_id'] for r in samples)
        id_classes = defaultdict(set)
        actual_contexts = defaultdict(set)
        for row in samples:
            id_classes[row['image_id']].add(row['class_name'])
            actual_contexts[row['class_name']].add(row['context'])
        declared = metadata['split_contexts'][split]
        if set(declared) != set(CLASS_NAMES):
            raise ValueError(f'Incorrect class list in {split} context metadata')
        # Empty context directories, if any, remain part of the saved protocol.
        for name in CLASS_NAMES:
            if not actual_contexts[name].issubset(set(declared[name])):
                raise ValueError(f'Undeclared context in {split}/{name}')
        contexts = {context for values in declared.values() for context in values}
        stats[split] = {
            'n_rows': len(samples), 'unique_image_ids': len(id_counts),
            'class_context_groups': sum(len(v) for v in declared.values()),
            'unique_context_tags': len(contexts),
            'per_class_rows': {c: counts[c] for c in CLASS_NAMES},
            'repeated_image_ids': sum(n > 1 for n in id_counts.values()),
            'multi_class_image_ids': sum(len(v) > 1 for v in id_classes.values()),
        }
        for key, value in stats[split].items():
            if value != metadata['splits'][split][key]:
                raise ValueError(f'{split} {key}: got {value}, expected {metadata["splits"][split][key]}')
        id_sets[split] = set(id_counts)
        context_sets[split] = contexts
    overlap = id_sets['train'] & id_sets['test']
    if overlap:
        raise ValueError(f'Train/test image-ID overlap: {len(overlap)}')
    for name in CLASS_NAMES:
        if set(metadata['split_contexts']['train'][name]) & set(metadata['split_contexts']['test'][name]):
            raise ValueError(f'Train/test assigned-context overlap for class {name}')
    global_overlap = len(context_sets['train'] & context_sets['test'])
    if global_overlap != metadata['global_context_tag_overlap']:
        raise ValueError('Global context overlap does not match release metadata')
    report = {'status': 'PASS', 'version': metadata['version'], 'splits': stats,
              'train_test_image_id_overlap': 0, 'global_context_tag_overlap': global_overlap,
              'checked_prepared_files': 0, 'checked_source_files': 0,
              'hashed_prepared_files': 0, 'hashed_source_files': 0}
    if dataset_root is not None:
        root = Path(dataset_root)
        expected_paths = {r['relative_path'] for rs in rows.values() for r in rs}
        actual_paths = {p.relative_to(root).as_posix() for split in rows
                        for p in (root/split).rglob('*') if p.is_file() or p.is_symlink()}
        if actual_paths != expected_paths:
            missing, extra = expected_paths-actual_paths, actual_paths-expected_paths
            raise ValueError(f'Dataset membership mismatch: {len(missing)} missing, {len(extra)} extra; '
                             f'examples missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}')
        expected_dirs = {f'{s}/{c}' for s in rows for c in CLASS_NAMES}
        expected_dirs.update(f'{s}/{c}/{c}({ctx})' for s in rows for c in CLASS_NAMES
                             for ctx in metadata['split_contexts'][s][c])
        actual_dirs = {p.relative_to(root).as_posix() for s in rows
                       for p in (root/s).rglob('*') if p.is_dir()}
        if actual_dirs != expected_dirs:
            raise ValueError('Dataset class/context directories differ from the frozen split')
        for samples in rows.values():
            for row in samples:
                path = root / row['relative_path']
                if not path.is_file():
                    raise FileNotFoundError(path)
                if check_hashes and sha256_file(path) != row['image_sha256']:
                    raise ValueError(f'Image checksum mismatch: {path}')
        report['checked_prepared_files'] = len(expected_paths)
        report['hashed_prepared_files'] = len(expected_paths) if check_hashes else 0
    if images_root is not None:
        root = Path(images_root)
        unique = {r['image_id']: r for rs in rows.values() for r in rs}
        for image_id, row in unique.items():
            path = root / f'{image_id}.jpg'
            if not path.is_file():
                raise FileNotFoundError(f'Missing upstream image {path}; use the folder containing <ID>.jpg files.')
            if check_hashes and sha256_file(path) != row['image_sha256']:
                raise ValueError(f'Image checksum mismatch: {path}')
        report['checked_source_files'] = len(unique)
        report['hashed_source_files'] = len(unique) if check_hashes else 0
    if check_hashes and dataset_root is None and images_root is None:
        raise ValueError('--check-hashes requires --dataset-root or --images-root')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--images-root', type=Path)
    parser.add_argument('--check-hashes', action='store_true', help='Compare every image with its original SHA-256.')
    parser.add_argument('--report', type=Path, help='Write a JSON verification report.')
    args = parser.parse_args()
    try:
        report = verify(args.manifest_dir, args.dataset_root, args.images_root, args.check_hashes)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'FAIL: {error}\n')
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
