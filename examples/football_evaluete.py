# +
from grf_imitation.infrastructure.execution.evaluate import create_parser, run, evaluate
from grf_imitation.user_config import LOCAL_DIR

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

parser = create_parser()
args = parser.parse_args([
    '--exp-dir',
    'data/gail_malib-5vs5/gail_malib-5vs5_0',
    '--track-progress',
    '--episodes',
    '10',
    '--render'
])
print(vars(args))
run(args)
