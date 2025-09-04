from pathlib import Path

import environ


def get_env() -> environ.Env:
    env_dir = Path(__file__).parent.parent

    _env = environ.Env()
    # _env.read_env("/.build.env")  # due https://github.com/moby/moby/issues/29110
    _env.read_env(str(env_dir / ".env"))
    return _env  # noqa: RET504


env = get_env()
