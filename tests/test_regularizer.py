# +
import matplotlib.pyplot as plt
import numpy as np

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

def gail_regularizer(x):
    assert np.all(x < 0), "x must less than 0, or will be positive infinity"
    return -x - np.log(1 - np.exp(x))


def plot_gail_regularizer():
    x = np.linspace(-10, -0.000000001, 200)
    g_x = gail_regularizer(x)
    fig, ax = plt.subplots(1,1)
    ax.plot(x, g_x, lw=3)
    ax.set_xlabel("cost")
    ax.set_ylabel("penalty value")
plot_gail_regularizer()


