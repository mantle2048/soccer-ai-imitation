"""
File taken from RLKit (https://github.com/vitchyr/rlkit).
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import json
import pickle
import errno
import time
import torch
import tempfile
import shelve

from typing import Dict, Tuple, List
from .tabulate import tabulate
from grf_imitation import user_config as conf

class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Logger(object):
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._video_log_dir = None

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

        self._shelf = None
        self._dump_cnt = 0

    @property
    def exp_dir(self):
        return self._snapshot_dir

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='w')

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def log(self, s, end='\n', with_prefix=True, with_timestamp=False):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out, end=end)
            for fd in list(self._text_fds.values()):
                fd.write(out + end)
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), val))

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """

        # Nothing to save
        if not data: return

        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, file_name, variant_data, mode='a'):
        self.log(f"{file_name}:", with_prefix=False)
        self.log(f"{variant_data}", with_prefix=False)
        file_name = osp.join(self._snapshot_dir, file_name)
        mkdir_p(os.path.dirname(file_name))
        with open(file_name, mode) as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='front'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
            # self.record_tabular(prefix + "Median" + suffix, np.median(vlues))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)
            # self.record_tabular(prefix + "Median" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)

        if self._dump_cnt == 0:
            self._dump_zero_tabular(*args, **kwargs)

        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                ### print tabulate
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]
        self._dump_cnt += 1

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params):
        if self._shelf is None:
            # exp_dir/run_dir/params/(params.bak | params.dat | params.dir)
            param_dir = osp.join(self._snapshot_dir, 'params')
            os.makedirs(param_dir, exist_ok=True)
            self._shelf = shelve.open(osp.join(param_dir, 'params'))
        self._shelf[str(itr)] = params
        self._shelf['last'] = params

    def close(self):
        self._shelf.close()

    def _dump_zero_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        zero_tabular = {}

        for key, val in dict(self._tabular).items():
            if type(val) == int:
                zero_tabular[key] = 0
            elif type(val) in (float, np.float32, np.float64):
                zero_tabular[key] = 0.
            elif type(val) == np.ndarray:
                zero_tabular[key] = np.array(0., dtype=np.float32)
            elif type(val) == np.float64:
                zero_tabular[key] = np.array(0., dtype=np.float32)
            else:
                raise ValueError()

        wh = kwargs.pop("write_header", None)
        for tabular_fd in list(self._tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd,
                                    fieldnames=list(zero_tabular.keys()))
            if wh or (
                    wh is None and tabular_fd not in self._tabular_header_written):
                writer.writeheader()
                self._tabular_header_written.add(tabular_fd)
            writer.writerow(zero_tabular)
            tabular_fd.flush()

    ########################################
    # Help Functions for Tensorboard & Aim #
    ########################################
    def log_scalar(self, scalar, name, step_):
        raise NotImplementedError

    def log_image(self, image, name, step_):
        raise NotImplementedError

    def log_video(self, video_frames, name, step, fps):
        raise NotImplementedError

    def log_figure(self, figure, name, step, phase):
        raise NotImplementedError

    def log_paths_as_videos(self, paths, step, max_videos_to_save, fps, video_title):
        raise NotImplementedError

