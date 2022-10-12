# Suppress numba debug logging
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('urllib').setLevel(logging.INFO)

from verticox.vantage6 import *
