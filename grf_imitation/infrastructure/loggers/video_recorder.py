""" yanked and modified from https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py"""
"""A wrapper for video recording environments by rolling it out, frame by frame."""
import json
import os
import os.path
import pkgutil
import shutil
import subprocess
import tempfile
from io import StringIO
from typing import List, Optional, Tuple, Union

import numpy as np

from gym import error, logger


def touch(path: str):
    """Touch a filename at path."""
    open(path, "a").close()


class VideoRecorder:  # TODO: remove with gym 1.0
    """VideoRecorder renders a nice movie of a rollout, frame by frame.
    It comes with an ``enabled`` option, so you can still use the same code on episodes where you don't want to record video.
    Note:
        You are responsible for calling :meth:`close` on a created VideoRecorder, or else you may leak an encoder process.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        render_fps: int = 30,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.
        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        self.enabled = enabled
        self._closed = False

        self.render_history = []
        self.last_frame = None

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(
                    suffix=required_ext, delete=False
                ) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        # Touch the file in any case, so we know it's present. This corrects for platform platform differences.
        # Using ffmpeg on OS X, the file is precreated, but not on Linux.
        touch(path)

        self.frames_per_sec = render_fps
        self.output_frames_per_sec = self.frames_per_sec

        self.encoder: Optional[
            Union[TextEncoder, ImageEncoder]
        ] = None  # lazily start the process
        self.broken = False

        self.empty = True

    @property
    def functional(self):
        """Returns if the video recorder is functional, is enabled and not broken."""
        return self.enabled and not self.broken

    def capture_frame(self, frame: Union[np.ndarray, List]):
        """Render the given `env` and add the resulting frame to the video."""
        if isinstance(frame, List):
            self.render_history += frame
            frame = frame[-1]
        self.last_frame = frame

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frame is None:
            # Indicates a bug in the environment: don't want to raise
            # an error here.
            logger.warn(
                "No captured frame. Disabling further video recorder"
            )
            self.broken = True
        else:
            self._encode_image_frame(frame)

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        if self.encoder:
            logger.debug("Closing video encoder: path=%s", self.path)
            self.encoder.close()
            self.encoder = None
        else:
            # No frames captured. remove the empty output file.
            os.remove(self.path)

        # If broken, get rid of the output file, otherwise we'd leak it.
        if self.broken:
            logger.info(
                "Cleaning up paths for broken video recorder: path=%s",
                self.path,
            )

            # Might have crashed before even starting the output file, don't try to remove in that case.
            if os.path.exists(self.path):
                os.remove(self.path)

        # Stop tracking this for autoclose
        self._closed = True

    def __del__(self):
        """Closes the environment correctly when the recorder is deleted."""
        # Make sure we've closed up shop when garbage collecting
        self.close()

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(
                self.path, frame.shape, self.frames_per_sec, self.output_frames_per_sec
            )

        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warn("Tried to pass invalid video frame, marking as broken: %s", e)
            self.broken = True
        else:
            self.empty = False

class ImageEncoder:
    """Captures image based frames of environments for Video Recorder."""

    def __init__(
        self,
        output_path: str,
        frame_shape: Tuple[int, int, int],
        frames_per_sec: int,
        output_frames_per_sec: int,
    ):
        """Encoder for capturing image based frames of environment for Video Recorder.
        Args:
            output_path: The output data path
            frame_shape: The expected frame shape, a tuple of height, weight and channels (3 or 4)
            frames_per_sec: The number of frames per second the environment runs at
            output_frames_per_sec: The output number of frames per second for the video
        Raises:
            InvalidFrame: Expects frame to have shape (w,h,3) or (w,h,4)
            DependencyNotInstalled: Found neither the ffmpeg nor avconv executables.
        """
        self.proc: Optional[subprocess.Popen] = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame(
                f"Your frame has shape {frame_shape}, but we require (w,h,3) or (w,h,4), "
                "i.e., RGB values for a w-by-h image, with an optional alpha channel."
            )
        self.wh = (w, h)
        self.includes_alpha = pixfmt == 4
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if shutil.which("avconv") is not None:
            self.backend = "avconv"
        elif shutil.which("ffmpeg") is not None:
            self.backend = "ffmpeg"
        elif pkgutil.find_loader("imageio_ffmpeg"):
            import imageio_ffmpeg

            self.backend = imageio_ffmpeg.get_ffmpeg_exe()
        else:
            raise error.DependencyNotInstalled(
                "Found neither the ffmpeg nor avconv executables. "
                "On OS X, you can install ffmpeg via `brew install ffmpeg`. "
                "On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. "
                "On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`. "
                "Alternatively, please install imageio-ffmpeg with `pip install imageio-ffmpeg`"
            )

        self.start()

    @property
    def version_info(self):
        """Returns the version info: backend, version and cmdline."""
        return {
            "backend": self.backend,
            "version": str(
                subprocess.check_output(
                    [self.backend, "-version"], stderr=subprocess.STDOUT
                )
            ),
            "cmdline": self.cmdline,
        }

    def start(self):
        """Starts a subprocess using the backend and cmdline."""
        self.cmdline = (
            self.backend,
            "-nostats",
            "-loglevel",
            "error",  # suppress warnings
            "-y",
            # input
            "-f",
            "rawvideo",
            "-s:v",
            "{}x{}".format(*self.wh),
            "-pix_fmt",
            ("rgb32" if self.includes_alpha else "rgb24"),
            "-framerate",
            "%d" % self.frames_per_sec,
            "-i",
            "-",  # this used to be /dev/stdin, which is not Windows-friendly
            # output
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "%d" % self.output_frames_per_sec,
            self.output_path,
        )

        logger.debug('Starting %s with "%s"', self.backend, " ".join(self.cmdline))
        if hasattr(os, "setsid"):  # setsid not present on Windows
            self.proc = subprocess.Popen(
                self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
            )
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame: Union[np.ndarray, np.generic]):
        """Captures a frame writing it to the backend subprocess."""
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                f"Wrong type {type(frame)} for {frame} (must be np.ndarray or np.generic)"
            )
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                f"Your frame has shape {frame.shape}, but the VideoRecorder is configured for shape {self.frame_shape}."
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                f"Your frame has data type {frame.dtype}, but we require uint8 (i.e. RGB values from 0-255)."
            )

        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write(frame.tobytes())
        except Exception:
            stdout, stderr = self.proc.communicate()
            logger.error("VideoRecorder encoder failed: %s", stderr)

    def close(self):
        """Closes the Image encoder."""
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error(f"VideoRecorder encoder exited with status {ret}")
