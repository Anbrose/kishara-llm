#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

docker = "docker"
docker_build = [docker, "buildx", "build", "--platform=linux/amd64"]
package_root = Path(__file__).parent.parent  # py-backend folder
tag_base = "registry.kishara.com.au/kishara-llm:latest"
tag_app = "registry.kishara.come.au/kishara-llm:latest"


def main():
    subprocess.run(
        docker_build
        + [
            "-f",
            (package_root / "scripts/Dockerfile-base"),
            "-t",
            tag_base,
            package_root,
        ],
        check=True,
    )
    subprocess.run(
        docker_build
        + ["-f", (package_root / "scripts/Dockerfile"), "-t", tag_app, package_root],
        check=True,
    )
    if "--push" in sys.argv[1:]:
        for tag in [tag_base, tag_app]:
            subprocess.run([docker, "push", tag], check=True)


if __name__ == "__main__":
    main()
