import numpy as np
import ffmpeg

__all__ = ['ReadVideoFFMPEG', 'WriteVideoFFMPEG']


class ReadVideoFFMPEG:

    def __init__(self, filename):
        self.filename = filename
        self._get_info()
        self._setup_process()

    def read_frame(self):
        frame_bytes = self.process.stdout.read(self.width * self.height * 3)
        frame = (
            np.frombuffer(frame_bytes, np.uint8)
            .reshape([self.height, self.width, 3]))
        return frame

    def read_frame_bytes(self):
        frame_bytes = self.process.stdout.read(self.width * self.height * 3)
        return frame_bytes

    def _setup_process(self):
        self.process = (
            ffmpeg
            .input(self.filename)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=False)
            )

    def _get_info(self):
        probe = ffmpeg.probe(self.filename)
        video_info = next(
            s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.num_frames = int(video_info['nb_frames'])


class WriteVideoFFMPEG:

    def __init__(self, filename, speed='superfast', bitrate='LOW4K', framerate=50.0):
        self.filename = filename
        self.frame_no = 0
        bitrates = {
            'LOW4K': '20000k',
            'MEDIUM4K': '50000k',
            'HIGH4K': '100000k',
            'LOW1080': '5000k',
            'MEDIUM1080': '10000k',
            'HIGH1080': '20000k'}
        self.video_bitrate = bitrates[bitrate]
        self.framerate=framerate
        self.preset = speed

    def add_frame(self, frame):
        if self.frame_no == 0:
            width = np.shape(frame)[1]
            height = np.shape(frame)[0]
            self._setup_process(width, height)
        self.process.stdin.write(frame.astype(np.uint8).tobytes())
        self.frame_no += 1

    def add_frame_bytes(self, frame, width, height):
        if self.frame_no == 0:
            self._setup_process(width, height)
        self.process.stdin.write(frame)
        self.frame_no += 1

    def _setup_process(self, width, height):
        self.process = (
            ffmpeg
            .input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                s='{}x{}'.format(width, height),
                r=50
            )
            .output(
                self.filename,
                pix_fmt='yuv420p',
                vcodec='libx264',
                preset=self.preset,
                video_bitrate=self.video_bitrate,
                r=self.framerate
            )
            .overwrite_output()
            .run_async(
                pipe_stdin=True,
                quiet=False
            )
        )

    def close(self):
        self.process.stdin.close()
        self.process.wait()

if __name__ == "__main__":
    from Generic import images
    input_video = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.mp4'
    vid = ReadVideoFFMPEG(input_video)
    for f in range(10):
        frame = vid.read_frame()
        images.display(frame)
