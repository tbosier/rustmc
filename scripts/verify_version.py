#!/usr/bin/env python3
"""Check release versions without dependencies, including on Python 3.9/3.10.

This reads only literal version/name/path fields in the repository's known TOML
layouts. It deliberately rejects unexpected forms instead of emulating a general
TOML parser; Cargo separately validates full manifests and lock resolution.
"""
import os
from pathlib import Path
import re
import sys


def scalar(text, key):
    matches = re.findall(r'^\s*' + re.escape(key) + r'\s*=\s*"([^"\n]+)"\s*(?:#.*)?$', text, re.M)
    if len(matches) != 1:
        raise ValueError("expected exactly one literal %s field" % key)
    return matches[0]


def table(path, name):
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r'^\[' + re.escape(name) + r'\]\s*\n(.*?)(?=^\[|\Z)', text, re.M | re.S)
    if len(matches) != 1:
        raise ValueError("%s must have exactly one [%s] table" % (path, name))
    return matches[0]


def verify(root, tag=None):
    versions = {
        "rust_core/Cargo.toml": scalar(table(root / "rust_core/Cargo.toml", "package"), "version"),
        "python_bindings/Cargo.toml": scalar(table(root / "python_bindings/Cargo.toml", "package"), "version"),
        "pyproject.toml": scalar(table(root / "pyproject.toml", "project"), "version"),
    }
    expected = versions["rust_core/Cargo.toml"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", expected):
        raise ValueError("release version must have the form X.Y.Z")
    if tag is not None:
        if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            raise ValueError("release tag must have the form vX.Y.Z")
        expected = tag[1:]
    dependencies = table(root / "python_bindings/Cargo.toml", "dependencies")
    matches = re.findall(r'^rustmc_core\s*=\s*\{([^}]+)\}\s*$', dependencies, re.M)
    if len(matches) != 1:
        raise ValueError("expected one inline rustmc_core path/version dependency")
    fields = dict(re.findall(r'(\w+)\s*=\s*"([^"\n]+)"', matches[0]))
    versions["python_bindings dependency rustmc_core"] = fields.get("version", "missing")
    if (root / "python_bindings" / fields.get("path", "")).resolve() != (root / "rust_core").resolve():
        raise ValueError("binding dependency must point to this workspace's rust_core")
    blocks = re.split(r'^\[\[package\]\]\s*$', (root / "Cargo.lock").read_text(encoding="utf-8"), flags=re.M)[1:]
    for name in ("rustmc", "rustmc_core"):
        packages = [block for block in blocks if re.search(r'^name\s*=\s*"' + name + r'"\s*$', block, re.M)]
        if len(packages) != 1:
            raise ValueError("Cargo.lock must contain exactly one %s package" % name)
        versions["Cargo.lock %s" % name] = scalar(packages[0], "version")
        if re.search(r'^source\s*=', packages[0], re.M):
            raise ValueError("Cargo.lock %s must resolve to the workspace package" % name)
    mismatches = ["%s=%s" % item for item in versions.items() if item[1] != expected]
    if mismatches:
        raise ValueError("expected version %s; mismatches: %s" % (expected, ", ".join(mismatches)))
    return expected


def main():
    if len(sys.argv) > 2:
        raise ValueError("usage: verify_version.py [vX.Y.Z]")
    reference = os.environ.get("GITHUB_REF", "")
    tag = sys.argv[1] if len(sys.argv) == 2 else (reference[len("refs/tags/"):] if reference.startswith("refs/tags/") else None)
    version = verify(Path(__file__).resolve().parents[1], tag)
    print("Version OK: %s (manifests, binding dependency, Cargo.lock%s)" % (version, ", tag" if tag else ""))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print("Version check failed: %s" % error, file=sys.stderr)
        sys.exit(1)
