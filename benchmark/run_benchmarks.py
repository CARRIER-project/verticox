import os
from pathlib import Path

from python_on_whales import docker

_COMPOSE_FILE = Path(__file__).parent


def main():
    os.chdir(_COMPOSE_FILE)
    docker.compose.up(force_recreate=True)


if __name__ == "__main__":
    main()
