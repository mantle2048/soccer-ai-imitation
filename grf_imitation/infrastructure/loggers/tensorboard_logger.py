import os
from grf_imitation.infrastructure.loggers.base_logger import Logger
from tensorboardX import SummaryWriter
from moviepy import editor as mpy # this line for 'moviepy' correctly run
import numpy as np

class TensorBoardLogger(Logger):
    def __init__(self):
        super().__init__()
        self._summ_writer = None

    def set_snapshot_dir(self, dir_name):
        ''' set snapshot dir and init tensorboard writer '''
        self._snapshot_dir = dir_name
        self._summ_writer = SummaryWriter(self._snapshot_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        '''  only avaiable for the data saved by add_scalars !!! '''
        log_path = os.path.join(self._snapshot_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()

    def dump_tabular(self, *args, **kwargs):

        tabular_dict = dict(self._tabular)
        assert 'Itr' in tabular_dict.keys(), 'no item **Itr** in tabular for tensorboard logger'
        itr = tabular_dict.pop('Itr')
        for key, value in tabular_dict.items():
            self.log_scalar(value, key, itr)

        super().dump_tabular(*args, **kwargs)

        self.flush()

    def close(self):
        self._summ_writer.close()

