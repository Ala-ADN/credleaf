"""Export locked pipeline dependencies for a CPU-only container runtime.

The project lock also contains CUDA packages for Linux. Exclude those explicit
pins so uv can use the CPU torch wheel's dependency metadata instead.
"""
from pathlib import Path
import subprocess
import sys
import tomllib


def export(project: Path, destination: Path):
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    args = ["uv", "export", "--project", str(project), "--frozen", "--group", "dev",
            "--no-emit-project", "--no-hashes", "--no-annotate", "--no-header"]
    for package in lock["package"]:
        if package["name"].startswith("nvidia-") or package["name"] == "triton":
            args += ["--no-emit-package", package["name"]]
    with destination.open("w", encoding="utf-8") as target:
        subprocess.run(args, stdout=target, check=True)


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
