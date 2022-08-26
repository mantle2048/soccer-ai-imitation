import gym

def get_max_episode_steps(env):

    if hasattr(env, 'spec') and hasattr(env.spec, 'max_episode_steps'):
        return env.spec.max_episode_steps

    elif hasattr(env, 'config'):
        return env.config['max_episode_steps']

    elif hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps

    elif hasattr(env.unwrapped, '_max_episode_steps'):
        return env.unwrapped._max_episode_steps

    elif hasattr(env, 'max_episode_steps'):
        return env.max_episode_steps
    else:
        raise ValueError(f"Not found the max episode steps of given env {str(env)}")
