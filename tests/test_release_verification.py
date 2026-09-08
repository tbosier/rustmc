"""Release version drift must fail before an upload, on Python 3.9+ too."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_version_check", ROOT / "scripts/verify_version.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def release_copy(tmp_path):
    for name in ("rust_core/Cargo.toml", "python_bindings/Cargo.toml", "pyproject.toml", "Cargo.lock"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path


def test_versions_and_tags_are_checked_without_toml_dependency(tmp_path):
    root = release_copy(tmp_path)
    version = CHECK.verify(root)
    assert CHECK.verify(root, "v" + version) == version
    for invalid in ("v999.0.0", "not-a-tag", "v@CURRENT@-rc1"):
        with pytest.raises(ValueError):
            CHECK.verify(root, invalid)


@pytest.mark.parametrize("name,old,new", [
    ("pyproject.toml", 'version = "@CURRENT@"', 'version = "999.0.0"'),
    ("python_bindings/Cargo.toml", 'path = "../rust_core", version = "@CURRENT@"', 'path = "../rust_core", version = "999.0.0"'),
    ("python_bindings/Cargo.toml", 'path = "../rust_core"', 'path = "../elsewhere"'),
    ("Cargo.lock", 'name = "rustmc_core"\nversion = "@CURRENT@"', 'name = "rustmc_core"\nversion = "999.0.0"'),
    ("Cargo.lock", 'name = "rustmc"\nversion = "@CURRENT@"', 'name = "rustmc"\nversion = "999.0.0"'),
])
def test_dependency_and_lock_drift_are_rejected(tmp_path, name, old, new):
    root = release_copy(tmp_path)
    target = root / name
    old = old.replace("@CURRENT@", CHECK.verify(root))
    assert old in target.read_text()
    target.write_text(target.read_text().replace(old, new, 1))
    with pytest.raises(ValueError):
        CHECK.verify(root)


def test_version_cli_runs_from_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_version.py")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "binding dependency, Cargo.lock" in result.stdout


def test_artifact_gate_rejects_stale_extra_archives_before_install(tmp_path):
    version = CHECK.verify(ROOT)
    (tmp_path / ("rustmc-%s-cp39-abi3-manylinux_2_17_x86_64.whl" % version)).touch()
    (tmp_path / "unrelated-1.0-cp39-abi3-manylinux_2_17_x86_64.whl").touch()
    result = subprocess.run([sys.executable, str(ROOT / "scripts/test_release_artifact.py"), "wheel", "--dist-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "unrelated files" in result.stderr
