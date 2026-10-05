"""Retained digest for native evaluation policy evidence."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()



if __name__ == '__main__':
    raise SystemExit('native_preflight_eval execution is retired; only digest remains supported')
