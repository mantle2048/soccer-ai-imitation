import numpy as np
import os
from grf_imitation.infrastructure.utils import utils
from grf_imitation.envs import make_env
from grf_imitation.envs.feature_encoder import MyFeatureEncoder


if __name__ == '__main__':
    env = make_env(env_config={'opponent': 'random'})
    obs = env.reset()
    # img_obss = []
    # img_obss.append(env.render(mode='rgb_array'))
    ep_len = 0
    for i in range(3000):
        act = np.random.randint(0, 19, (4,))
        next_obs, rew, done, info = env.step(act)
        ep_len += 1
        # img_obss.append(env.render(mode='rgb_array'))
        if done:
            break
    # print(ep_len)
    # img_obss = np.array(img_obss)
    # utils.write_mp4('test', img_obss)
