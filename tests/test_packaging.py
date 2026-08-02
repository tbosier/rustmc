"""
Packaging validation: wheel contents, metadata, and (optionally) a full
build-and-install-into-a-clean-environment round trip.

Fast checks (no network, no build) run whenever a wheel is already present
in dist/ -- they skip with a clear reason otherwise, so `pytest tests/ -q`
stays hermetic by default.

The full round trip (`test_wheel_installs_in_clean_environment`) is marked
`network` and deselected by default (see `addopts` in pyproject.toml)
because it builds a wheel with maturin and creates a fresh venv, which
needs network access this sandbox only grants when explicitly
unsandboxed. CI exercises the equivalent path directly as dedicated
workflow jobs (see .github/workflows/ci.yml) using
scripts/verify_wheel_install.py, which is the same script this test
shells out to.
"""
import glob
import os
import re
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_wheel():
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_text).group(1)
    wheel_glob = f"rustmc-{version}-*.whl"
    matches = sorted(glob.glob(str(REPO_ROOT / "dist" / wheel_glob)))
    matches += sorted(glob.glob(str(REPO_ROOT / "target" / "wheels" / wheel_glob)))
    return matches[0] if matches else None


@pytest.fixture(scope="module")
def wheel_path():
    whl = _find_wheel()
    if whl is None:
        pytest.skip(
            "no built wheel found in dist/ or target/wheels/; "
            "run `maturin build --release` first to exercise packaging tests"
        )
    return whl


def test_wheel_contains_expected_files(wheel_path):
    with zipfile.ZipFile(wheel_path) as z:
        names = z.namelist()

    assert any(n.endswith(".so") or n.endswith(".pyd") for n in names), (
        f"wheel has no compiled extension module: {names}"
    )
    assert "rustmc/__init__.py" in names, f"wheel missing rustmc/__init__.py: {names}"
    assert any("dist-info/METADATA" in n for n in names), "wheel missing METADATA"
    assert any("dist-info/RECORD" in n for n in names), "wheel missing RECORD"
    assert any("LICENSE" in n for n in names), "wheel missing LICENSE"


def test_wheel_metadata_matches_pyproject(wheel_path):
    # Parsed with a regex rather than tomllib so this test works on Python
    # 3.9/3.10 too (tomllib is stdlib only from 3.11+), matching the
    # project's actual supported-version floor.
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    expected_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_text).group(1)
    expected_requires_python = re.search(
        r'(?m)^requires-python\s*=\s*"([^"]+)"', pyproject_text
    ).group(1)

    with zipfile.ZipFile(wheel_path) as z:
        metadata_name = next(n for n in z.namelist() if n.endswith("dist-info/METADATA"))
        metadata = z.read(metadata_name).decode()

    assert f"Version: {expected_version}" in metadata, metadata
    assert f"Requires-Python: {expected_requires_python}" in metadata, metadata
    assert "License-Expression: MIT" in metadata, metadata
    assert "License-File: LICENSE" in metadata, metadata
    assert (
        "Project-URL: Changelog, "
        "https://github.com/tbosier/rustmc/blob/main/CHANGELOG.md"
    ) in metadata
    assert "Project-URL: Documentation, https://tbosier.github.io/rustmc/" in metadata
    assert "Project-URL: Issues, https://github.com/tbosier/rustmc/issues" in metadata


def test_wheel_uses_python39_stable_abi(wheel_path):
    """One wheel per platform must import on every claimed CPython 3.9+ version."""
    wheel_name = Path(wheel_path).name
    assert "-cp39-abi3-" in wheel_name, (
        f"expected a CPython 3.9 stable-ABI wheel, got {wheel_name}"
    )


def test_release_versions_are_synchronized():
    manifests = [
        REPO_ROOT / "rust_core" / "Cargo.toml",
        REPO_ROOT / "python_bindings" / "Cargo.toml",
        REPO_ROOT / "pyproject.toml",
    ]
    versions = {}
    for manifest in manifests:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', manifest.read_text())
        assert match is not None, f"no package version found in {manifest}"
        versions[str(manifest.relative_to(REPO_ROOT))] = match.group(1)
    assert len(set(versions.values())) == 1, f"release version mismatch: {versions}"


def test_python_bridge_cannot_collide_on_crates_io():
    binding_manifest = (REPO_ROOT / "python_bindings" / "Cargo.toml").read_text()
    assert re.search(r'(?m)^publish\s*=\s*false\s*$', binding_manifest), (
        "the PyO3 bridge is named rustmc, which is occupied on crates.io; "
        "keep it publish=false and publish rustmc_core instead"
    )


@pytest.mark.xfail(strict=True, reason=(
    "known packaging gap: no py.typed marker or .pyi stubs are shipped, "
    "so type checkers see an untyped package"
))
def test_wheel_ships_type_information(wheel_path):
    with zipfile.ZipFile(wheel_path) as z:
        names = z.namelist()
    assert any(n.endswith("py.typed") for n in names) or any(n.endswith(".pyi") for n in names)


def test_no_stray_rustmc_shadow_at_repo_root():
    """
    Guard against a future accidental top-level rustmc/ directory or
    rustmc.py file at the repo root, which would shadow the installed
    package and reintroduce the exact sys.path leakage this task exists to
    prevent.
    """
    assert not (REPO_ROOT / "rustmc").exists()
    assert not (REPO_ROOT / "rustmc.py").exists()


@pytest.mark.network
def test_wheel_installs_in_clean_environment(tmp_path):
    """
    Full round trip: `maturin build`, install the resulting wheel into a
    brand-new venv created outside the repo (tmp_path, a pytest tmp
    directory under the system temp dir), and run
    scripts/verify_wheel_install.py from that same directory -- so the
    only `rustmc` importable is the one just installed from the wheel.
    """
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    subprocess.run(
        [
            sys.executable, "-m", "maturin", "build", "--release",
            "--out", str(dist_dir),
            "--manifest-path", str(REPO_ROOT / "python_bindings" / "Cargo.toml"),
        ],
        check=True,
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, "maturin build produced no wheel"

    env_dir = tmp_path / "clean-env"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    env_python = env_dir / "bin" / "python"

    subprocess.run(
        [str(env_python), "-m", "pip", "install", "--quiet", str(wheels[0]), "numpy"],
        check=True,
    )

    script_src = REPO_ROOT / "scripts" / "verify_wheel_install.py"
    script_dst = tmp_path / "verify_wheel_install.py"
    script_dst.write_text(script_src.read_text())

    result = subprocess.run(
        [str(env_python), str(script_dst)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "RUSTMC_REQUIRE_SITE_PACKAGES": "1"},
    )
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, "verify_wheel_install.py failed against the clean install"
    assert "site-packages" in result.stdout
    assert str(REPO_ROOT) not in result.stdout
