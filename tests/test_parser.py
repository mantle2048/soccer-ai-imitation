# +
import numpy as np
import argparse

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

from grf_imitation.infrastructure.evaluation.evaluate import create_parser

parser = create_parser()

args = parser.parse_args([
    'data/exp_dir',
    '--local-mode'
])

args


