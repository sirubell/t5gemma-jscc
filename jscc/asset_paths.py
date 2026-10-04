"""Local immutable asset paths and cheap acquisition trust signatures."""
from pathlib import Path


def normalize_asset_path(path):
    """Resolve a local path and canonicalize the approved /work mount alias."""
    value = Path(path).expanduser()
    if '..' in value.parts:
        raise ValueError('asset path traversal rejected')
    resolved = value.resolve()
    if resolved == Path('/work') or Path('/work') in resolved.parents:
        resolved = Path('/raid/work') / resolved.relative_to('/work')
    return resolved


def asset_file_path(root, name):
    """Allow regular contained files or HF snapshot links to its repository blobs."""
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or str(relative) != str(name):
        raise ValueError('asset filename traversal rejected')
    root = normalize_asset_path(root)
    logical = root / relative
    resolved = logical.resolve(strict=True)
    if resolved == logical:
        if not resolved.is_file():
            raise ValueError('asset must be a regular file')
        return resolved
    # Snapshot links are the only supported indirection. No directory symlinks
    # or links to another snapshot/repository are acquisition equivalents.
    if (root.parent.name != 'snapshots' or len(relative.parts) != 1
            or not logical.is_symlink() or resolved.parent != root.parent.parent / 'blobs'
            or not resolved.is_file()):
        raise ValueError('asset symlink target outside pinned HF repository blobs')
    return resolved


def asset_trust_signature(path):
    target = normalize_asset_path(path)
    stat = target.stat()
    return {'target': str(target), 'dev': stat.st_dev, 'inode': stat.st_ino,
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def verify_asset_record(path, record, trust=None):
    if not path.is_file():
        raise FileNotFoundError(f'recipe cached asset missing: {path}')
    if path.stat().st_size != record['bytes']:
        raise ValueError(f'cached asset size differs: {path}')
    if trust is not None and asset_trust_signature(path) != trust:
        raise ValueError(f'cached asset trust signature differs; reacquire or audit: {path}')


def asset_stat(path):
    """Public cheap immutable-file signature for preparation consumers."""
    return asset_trust_signature(path)


def verify_asset_stat(path, expected):
    """Reject changed immutable file state without reading its contents."""
    if asset_stat(path) != expected:
        raise ValueError(f'cached asset trust signature differs; reacquire or audit: {path}')


def tokenizer_asset_names(root):
    """Names consumed by the supported local tokenizer/config cache."""
    return sorted({p.name for pattern in ('config.json', 'tokenizer*', '*token*.json', '*.model')
                   for p in Path(root).glob(pattern) if p.is_file()})
