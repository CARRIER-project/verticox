import os
import subprocess

from vantage6.common import info

DEFAULT_PORT = 8888


def RPC_init(data, *args, **kwargs):
    info('Starting java server')
    # TODO: Create properties file
    subprocess.run(['java', '-jar', _get_jar_path()])


def _get_jar_path():
    return os.environ.get('JAR_PATH')


def init_local():
    process = subprocess.Popen(['java', '-jar', _get_jar_path()])

    return process
