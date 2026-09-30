"""Portable, standard-library-only access to the frozen MetaShift Animals split."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

CLASS_NAMES = ('bear', 'bird', 'cow', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra')
RELEASE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = RELEASE_ROOT / 'manifests'
FIELDS = ('row_id', 'split', 'image_id', 'class_id', 'class_name', 'context',
          'relative_path', 'image_sha256')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(manifest_dir=DEFAULT_MANIFEST_DIR) -> dict:
    path = Path(manifest_dir).parent / 'metadata.json'
    with path.open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if metadata.get('schema_version') != 1 or tuple(metadata.get('class_names', [])) != CLASS_NAMES:
        raise ValueError(f'Unsupported dataset metadata: {path}')
    return metadata


def load_manifest(split: str, manifest_dir=DEFAULT_MANIFEST_DIR) -> list[dict]:
    """Read canonical rows in experiment order; verify their checksum and schema."""
    if split not in ('train', 'test'):
        raise ValueError("MetaShift Animals has only 'train' and 'test'; no independent validation split.")
    manifest_dir = Path(manifest_dir)
    metadata = load_metadata(manifest_dir)
    path = manifest_dir / f'{split}.csv'
    if sha256_file(path) != metadata['splits'][split]['manifest_sha256']:
        raise ValueError(f'Manifest checksum mismatch: {path}')
    rows = []
    seen_paths = set()
    image_hashes = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f'Unexpected manifest columns: {path}')
        for index, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV record in {path}, row {index}')
            row['row_id'] = int(row['row_id'])
            row['class_id'] = int(row['class_id'])
            label = row['class_id']
            if row['row_id'] != index or row['split'] != split:
                raise ValueError(f'Invalid row ordering/split in {path}, row {index}')
            if not 0 <= label < len(CLASS_NAMES) or row['class_name'] != CLASS_NAMES[label]:
                raise ValueError(f'Invalid class mapping in {path}, row {index}')
            if not re.fullmatch(r'[0-9]+', row['image_id']):
                raise ValueError(f'Invalid image ID in {path}, row {index}')
            context = row['context']
            if not context or any(c in context for c in ('/', '\\', '\x00', '\n', '\r')):
                raise ValueError(f'Invalid context in {path}, row {index}')
            expected = f"{split}/{row['class_name']}/{row['class_name']}({context})/{row['image_id']}.jpg"
            if row['relative_path'] != expected or expected in seen_paths:
                raise ValueError(f'Invalid or repeated sample path in {path}, row {index}')
            if not re.fullmatch(r'[0-9a-f]{64}', row['image_sha256']):
                raise ValueError(f'Invalid image hash in {path}, row {index}')
            old_hash = image_hashes.setdefault(row['image_id'], row['image_sha256'])
            if old_hash != row['image_sha256']:
                raise ValueError(f'Conflicting content hashes for image {row["image_id"]}')
            seen_paths.add(expected)
            rows.append(row)
    if len(rows) != metadata['splits'][split]['n_rows']:
        raise ValueError(f'Wrong number of rows: {path}')
    return rows
