import os
import time
import copy
import PIL
import numpy as np

from grf_imitation import user_config as conf

from os import path as osp
from moviepy.editor import ImageSequenceClip, VideoClip

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['obs']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################


def standardize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def de_standardize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

############################################
############################################

def save_video(dir, name, frames):
    if frames.size == 0: return
    video_dir = osp.join(dir, 'videos') 
    os.makedirs(video_dir, exist_ok=True)
    video_path = osp.join(video_dir, name) 
    write_mp4(video_path, frames, **conf.LOCAL_GIF_CONFIG)

def write_mp4(filename, array, fps=20, scale=1.0, backend='MoviePy'):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64, 3)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    if backend == 'PIL':
        # Low quality but quick gif save by using PIL
        imgs = [PIL.Image.fromarray(img) for img in array]
        imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)
        return filename

    elif backend == 'MoviePy':
        # High quality but slow gif save by using moviepy
        clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
        # clip.write_gif(filename, program='ffmpeg', fps=fps)
        clip.write_videofile(filename, fps=fps)
        return clip

    else:
        raise ValueError
