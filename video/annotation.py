from tqdm import tqdm
import os
from Generic import images
import pygame
from Generic import video


__all__ = ['ExampleChild', 'Annotator']

class Annotator:

    def __init__(self, in_name, out_name, frames_as_surface=False):
        in_name = self.check_crop(in_name)
        self.open_video(in_name)
        self.output_filename = out_name
        self.frames_as_surface = frames_as_surface


    def open_video(self, in_name):
        self.cap = video.ReadVideoFFMPEG(in_name)
        self.width = self.cap.width
        self.height = self.cap.height

    def annotate(self, bitrate='HIGH1080'):
        self.out = video.WriteVideoFFMPEG(self.output_filename,
                                          bitrate=bitrate)
        for f in tqdm(range(self.cap.num_frames), 'Annotation '+self.parameter):

            frame = self.read_frame()

            frame = self.process_frame(frame, f)

            self.write_frame(frame)
        self.out.close()

    def read_frame(self):
        if self.frames_as_surface:
            frame = self.cap.read_frame_bytes()
            return pygame.image.fromstring(
                frame, (self.cap.width, self.cap.height), 'RGB')
        else:
            return self.cap.read_frame()

    def write_frame(self, frame):
        if self.frames_as_surface:
            frame = pygame.image.tostring(frame, 'RGB')
            self.out.add_frame_bytes(frame, self.width, self.height)
        else:
            self.out.add_frame(frame)

    def check_crop(self, filename):
        return filename

    def process_frame(self, frame, f):
        return frame


class ExampleChild(Annotator):
    def __init__(self,
                 filename):
        input_filename = filename
        output_filename = os.path.splitext(filename)[0] + '_ex.mp4'
        Annotator.__init__(self, input_filename, output_filename)

    def annotate_frame(self, frame, f):
        frame = images.draw_circle(frame, 20, 20, 10, thickness=-1)
        return frame


if __name__ == "__main__":
    from Generic import filedialogs, video
    file = filedialogs.load_filename()
    ec = video.ExampleChild(file)
    ec.annotate()