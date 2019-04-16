from . import *
from tqdm import tqdm
import os
import pygame


class Annotator:

    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename

    def annotate(self, bitrate='MEDIUM4K', read_as_bytes=False):
        self.cap = ReadVideoFFMPEG(self.input_filename)
        out = WriteVideoFFMPEG(self.output_filename,
                               bitrate=bitrate)
        for f in tqdm(range(self.cap.num_frames), 'Annotation'):
            if read_as_bytes:
                frame = self.cap.read_frame_bytes()
            else:
                frame = self.cap.read_frame()
            frame = self.annotate_frame(frame, f)
            if read_as_bytes:
                out.add_frame_bytes(frame, self.cap.width, self.cap.height)
            else:
                out.add_frame(frame)
        out.close()

    def annotate_frame(self, frame, f):
        """Overwrite this method in child class"""
        return frame


class CircleAnnotate(Annotator):
    def __init__(self,
                 dataframe,
                 filename):
        self.td = dataframe
        self.filename = os.path.splitext(filename)[0]
        input_filename = self.filename + '_crop.mp4'
        output_filename = self.filename + '_circles.mp4'
        Annotator.__init__(input_filename, output_filename)

    def annotate_frame(self, frame, f):
        surface = pygame.image.fromstring(
            frame, (self.cap.width, self.cap.height), 'RGB')
        info = self.td.get_info(f, ['x', 'y', 'r', 'particle'])
        col = (255, 0, 0)
        for xi, yi, r, param in info:
            pygame.draw.circle(
                surface, col, (int(xi), int(yi)), int(r), 3)
        if self.shrink_factor != 1:
            surface = pygame.transform.scale(
                surface,
                (self.cap.width // self.shrink_factor,
                 self.cap.height // self.shrink_factor))
        frame = pygame.image.tostring(surface, 'RGB')
        return frame