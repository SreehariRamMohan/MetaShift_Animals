#!/usr/bin/env python3
"""Copy upstream MetaShift/GQA images into the exact eight-class benchmark split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from manifest import CLASS_NAMES, DEFAULT_MANIFEST_DIR, load_manifest, load_metadata, sha256_file
from verify import verify


def prepare(images_root: Path, output: Path, manifest_dir=DEFAULT_MANIFEST_DIR) -> dict:
    """Preflight all source bytes; copy, never move; refuse an existing destination."""
    images_root, output, manifest_dir = Path(images_root), Path(output), Path(manifest_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}. Choose a new output directory.')
    metadata = load_metadata(manifest_dir)
    rows = {s: load_manifest(s, manifest_dir) for s in ('train', 'test')}
    # Validate split semantics and every required original image before creating output.
    print('Checking frozen manifests and all 8,428 source image hashes...', flush=True)
    verify(manifest_dir, images_root=images_root, check_hashes=True)
    output.mkdir(parents=True, exist_ok=False)
    for split, samples in rows.items():
        for name in CLASS_NAMES:
            for context in metadata['split_contexts'][split][name]:
                (output / split / name / f'{name}({context})').mkdir(parents=True, exist_ok=True)
        for row in samples:
            source = images_root / f'{row["image_id"]}.jpg'
            destination = output / row['relative_path']
            shutil.copy2(source, destination)
            if sha256_file(destination) != row['image_sha256']:
                raise ValueError(f'Copied image checksum mismatch: {destination}')
        print(f'Copied {len(samples):,} {split} rows.', flush=True)
    (output/'manifests').mkdir()
    for split in rows:
        shutil.copy2(manifest_dir/f'{split}.csv', output/'manifests'/f'{split}.csv')
    shutil.copy2(manifest_dir.parent/'metadata.json', output/'metadata.json')
    # Each copied image was hashed above; final pass also checks complete membership.
    report = verify(manifest_dir, dataset_root=output)
    report['hashed_source_files'] = sum(v['unique_image_ids'] for v in report['splits'].values())
    report['checked_source_files'] = report['hashed_source_files']
    report['hashed_prepared_files'] = sum(v['n_rows'] for v in report['splits'].values())
    (output/'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(f'Ready: {output}\nAll copied images match the original content hashes.', flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-root', required=True, type=Path, help='MetaShift/GQA directory containing <ID>.jpg files.')
    parser.add_argument('--output', required=True, type=Path, help='New output directory; existing paths are never overwritten.')
    parser.add_argument('--manifest-dir', type=Path, default=DEFAULT_MANIFEST_DIR)
    args = parser.parse_args()
    try:
        prepare(args.images_root, args.output, args.manifest_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'FAIL: {error}\nIf copying was interrupted, the partial output is retained; choose a new output path.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
