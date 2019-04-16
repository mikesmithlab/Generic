from tqdm import tqdm
import os
from Generic import images
from Generic import video


__all__ = ['ExampleChild', 'Annotator']

class Annotator:

    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename

    def annotate(self, bitrate='HIGH1080'):
        self.cap = video.ReadVideoFFMPEG(self.input_filename)
        out = video.WriteVideoFFMPEG(self.output_filename,
                               bitrate=bitrate)
        for f in tqdm(range(self.cap.num_frames), 'Annotation'):
            frame = self.cap.read_frame()
            frame = self.annotate_frame(frame, f)
            out.add_frame(frame)
        out.close()

    def annotate_frame(self, frame, f):
        """Overwrite this method in child class"""
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