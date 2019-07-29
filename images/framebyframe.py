from Generic.video import ReadVideo
from Generic.images import display


if __name__ == '__main__':
    readvid = ReadVideo()

    for i in range(readvid.num_frames):
        frame = readvid.read_next_frame()
        display(frame)
    