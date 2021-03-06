import os

# import pims
import cv2
import numpy as np
from slicerator import Slicerator

from Generic import filedialogs as fd
from Generic import images

__all__ = ["WriteVideo", "ReadVideo", ]

class WriteVideo:
    '''
    Class is designed to make writing to video files easy.

    Keyword Arguments
    -----------------
    filename - Specifies output filename.
                    If filename == None then a dialog will ask for
                    output file to be supplied
    frame_size - Tuple (height,width,colour depth).
                    if frame = image must match the dimension of the image
                    if frame_size = None frame_size is set from the dimensions
                    of the supplied image.
                    N.B (h,w,d) is the opposite of cv2 which requires (w,h,d)
                    but this is the same as np.shape
    frame - supply frame from which frame_size is calculated. This can be used
            as first frame of output video
    write_frame - if True frame is used as first frame of video
    fps - framerate for the video
    codec - 'XVID'  -  'XVID' seems to work for  mp4 and avi
            Grayscale images are converted to pseudo colour before writing.


    Variables
    ---------
        self.filename - output filename
        self.frame_size - frame dimensions as tuple (w,h,colour depth)
        self.fps - framerate of output video
        self.write_vid - instance of cv2 Video Object

    Methods
    -------
    add_frame() - requires an image which matches dimensions of
                 self.frame_size
    close() - releases video object
    '''

    def __init__(self, filename=None, frame_size=None, frame=None,
                 write_frame=False, fps=30.0, codec='XVID'):

        # codec_list = ['XVID','mp4v']
        codec_code = list(codec)

        # Check inputs for errors
        if (frame_size is None) and (frame is None):
            raise ArgumentsMissing(('frame_size', 'frame'))
        if (frame_size is not None) and (frame is not None):
            if frame_size != np.shape(frame):
                raise ImageShapeError(frame, frame_size)

        # Optional arguments to get frame_size and set codec type
        if frame is None:
            self.frame_size = frame_size
        elif frame_size is None:
            frame_size = np.shape(frame)
            self.frame_size = frame_size

        fourcc = cv2.VideoWriter_fourcc(
            codec_code[0], codec_code[1],
            codec_code[2], codec_code[3]
        )

        if filename is None:
            filename = fd.save_filename(caption='select movie filename',
                                        file_filter='*.avi;;*.mp4')

        self.write_vid = cv2.VideoWriter(
            filename, fourcc, fps,
            (self.frame_size[1], self.frame_size[0]))

        if (write_frame is not None) and (frame is not None):
            self.add_frame(frame)
        self.fps = fps
        self.filename = filename
        if self.write_vid.isOpened():
            print('Video open for writing')
        else:
            print('Issue correctly opening video')
            print('Prob a compatability problem between codec '
                  'and format. Grayscale images are more fussy')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add_frame(self, img):
        if self.frame_size == np.shape(img):
            if np.size(self.frame_size) == 2:
                img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)

            self.write_vid.write(img)
        else:
            raise ImageShapeError(img, self.frame_size)

    def close(self):
        self.write_vid.release()
        # print('Video closed for writing')


@Slicerator.from_class
class ReadVideo:
    '''
    Class is designed to handle the reading of video.
    Videos can be read from .mp4 and .avi formats.

    Keyword Arguments
    -----------------
    filename: Specifies output file.
                    If filename == None then a dialog will ask for
                    output filename to be supplied

    Variables
    ---------
        self.filename - filename
        self.framenum - current frame
        self.read_vid - Instance of cv2 Video Object
        self.num_frames - Number of frames in video
        self.width - width of frame
        self.height - height of frame
        self.fps - Video framerate
        self.current_time - Current time in ms
                            (Not always supported)
        self.format - format of video (Not always supported)
        self.codec - codec of video

    Methods
    -------
        __init__(filename=None) - Create Video Object
        open_video() - opens Video for reading
        get_vid_props(show=True) - get video properties and print
                           to terminal if show==True

        read_next_frame() - read the next frame in video
        find_frame(frame_num) - find frame specified by framenum
        set_frame(frame_num) - sets the frame specified by framenum
        generate_frame_filename - creates an appropriate filename for an image
                                ie vid filename_00035.png where number digits
                                relates to total number of frames in video
        close() - closes Video for reading

    Example Usage
    -------------
        vid = ReadVideo()
        #Print out video properties to the terminal
        vid.get_vid_props(show=True)
        frame = vid.read_next_frame()
        frame = vid.find_frame(frame_num)
        vid.generate_frame_filename(frame_num)
        vid.close()
    '''

    def __init__(self, filename=None, grayscale=False, return_function=None):
        '''
        Initialise video reading object
        if filename = None user must select filename with dialogue
        '''
        if filename is None:
            filename = fd.load_filename(caption='select movie filename',
                                        file_filter='*.avi;;*.MP4;;*.mp4;;*.tif;;*.*')
        self.filename = filename
        self.grayscale = grayscale
        self.return_func = return_function
        self._detect_file_type()
        self.open_video()
        self.get_vid_props()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _detect_file_type(self):
        _, self.ext = os.path.splitext(self.filename)
        if self.ext in ['.MP4', '.mp4', '.m4v','.avi']:
            self.filetype = 'video'

        elif self.ext in ['.tif']:
            self.filetype = 'img_seq'
        else:
            print('sequence format not supported, '
                  'test and add to list if necessary in _detect_file_type')

    def open_video(self):
        '''Creates a video Object for reading'''
        self.frame_num = 0
        self.read_vid = cv2.VideoCapture(self.filename)

    def get_vid_props(self, show=False):


        self.frame_num = self.read_vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.num_frames = int(self.read_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_time = self.read_vid.get(cv2.CAP_PROP_POS_MSEC)
        self.width = int(self.read_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.read_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.read_vid.get(cv2.CAP_PROP_MONOCHROME) == 0.0:
            self.colour = 3
            self.frame_size=(self.height,self.width,3)
        else:
            self.colour = 1
            self.frame_size = (self.width, self.height)
        self.fps = self.read_vid.get(cv2.CAP_PROP_FPS)
        self.format = self.read_vid.get(cv2.CAP_PROP_FORMAT)
        self.codec = self.read_vid.get(cv2.CAP_PROP_FOURCC)

        self.file_extension = self.filename.split('.')[1]

        if show:
            print('----------------------------')
            print('List of Video Properties')
            print('----------------------------')
            print('file_type : ', self.filetype)
            print('frame_num : ', self.frame_num)
            print('num_frames : ', self.num_frames)
            print('current_time (ms) : ', self.current_time)
            print('width : ', self.width)
            print('height : ', self.height)
            print('colour: ', self.colour)
            print('fps : ', self.fps)
            print('format : ', self.format)
            print('codec : ', self.codec)
            print('file_extension :', self.file_extension)
            print('')
            print('unsupported features return 0')
            print('-----------------------------')

    def read_next_frame(self):
        '''reads the next available frame'''
        ret, img = self.read_vid.read()

        self.frame_num = self.frame_num + 1
        if ret:
            if self.grayscale:
                return images.bgr_2_grayscale(img)
            if self.return_func:
                return self.return_func(img)
            else:
                return img
        else:
            print('Error reading the frame. Check path and filename carefully')

    def __getitem__(self, item):
        return self.find_frame(item)

    def __len__(self):
        return self.num_frames

    def find_frame(self, frame_num):
        '''searches for specific frame and reads it'''
        self.set_frame(frame_num)
        img = self.read_next_frame()
        self.frame_num = frame_num
        return img

    def set_frame(self, frame_num):
        if self.filetype == 'video':
            """Moves the video reader to the given frame"""
            self.read_vid.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
        elif self.filetype == 'img_seq':
            self.frame_num = frame_num
        else:
            print('Error in set_frame')
        self.frame_num = frame_num

    def generate_frame_filename(self, ext='.png'):
        len_file_ext = len(self.file_extension) + 1
        filename_string = (
                self.filename[:-len_file_ext] + '_' +
                '0' * (len(str(int(self.num_frames))) -
                       len(str(int(self.frame_num)))) +
                str(int(self.frame_num)) + ext
        )
        return filename_string

    def close(self):
        if self.filetype == 'video':
            self.read_vid.release()
        elif self.filetype == 'img_seq':
            self.read_vid.close()


class ImageShapeError(Exception):
    '''
    Tries to helpfully work out what is wrong with supplied image
    It distinguishes between the image being empty and the image being the
    wrong way round. (width,height) rather than (height,width) as required

    Arguments:
    img is the image being used
    frame_size was the specified dimensions in the instance definition
    '''

    def __init__(self, img, frame_size):
        img_shape = np.shape(img)
        print('Image supplied ', img_shape)
        print('frame_size ', frame_size)
        if img_shape == (frame_size[1], frame_size[0], frame_size[2]):
            print('frame_size should be (h,w,d) not (w,h,d)')
        if img_shape[2] != frame_size[2]:
            print('check colour depth of image')


class ArgumentsMissing(Exception):
    '''
    Designed to inform user they are missing necessary arguments to instantiate
    a class
    '''

    def __init__(self, arguments_missing):
        print('You have failed to supply all necessary arguments')
        print('In this case: ', arguments_missing)