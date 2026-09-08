#!/usr/bin/env python3
"""Install one exact release artifact in a clean environment and run its tests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

from verify_version import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wheel", "sdist"))
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = verify(root)
    pattern = "rustmc-%s-*.whl" if args.kind == "wheel" else "rustmc-%s.tar.gz"
    artifacts = sorted(args.dist_dir.resolve().glob(pattern % expected))
    all_artifacts = sorted(args.dist_dir.resolve().glob("*.whl" if args.kind == "wheel" else "*.tar.gz"))
    if len(artifacts) != 1 or artifacts != all_artifacts:
        raise ValueError("expected only one %s artifact for %s, found %d matching artifacts or unrelated files" % (args.kind, expected, len(artifacts)))
    with tempfile.TemporaryDirectory(prefix="rustmc-release-test-") as directory:
        temporary = Path(directory)
        environment = temporary / "env"
        # Portable Python builds can use an executable-relative libpython path;
        # copying that executable into a venv loses the library location on Unix.
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(executable), "-m", "pip", "install", str(artifacts[0]), "numpy", "pytest"], check=True)
        # cwd and the import path stay outside the checkout. Tests can reference
        # repository fixtures while the import resolves to the installed archive.
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env["RUSTMC_REQUIRE_SITE_PACKAGES"] = "1"
        subprocess.run([str(executable), "-c", "import rustmc; assert rustmc.__version__ == %r" % expected], cwd=temporary, env=clean_env, check=True)
        subprocess.run([str(executable), str(root / "scripts/verify_wheel_install.py")], cwd=temporary, env=clean_env, check=True)
        subprocess.run([str(executable), "-m", "pytest", "-q", str(root / "tests")], cwd=temporary, env=clean_env, check=True)
    print("Verified exact release artifact: %s" % artifacts[0])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("Release artifact verification failed: %s" % error, file=sys.stderr)
        sys.exit(1)
