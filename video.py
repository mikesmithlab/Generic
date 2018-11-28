import numpy as np
import cv2
import Generic.filedialogs as fd
import subprocess
import os
import ffmpeg


def crop_video(filename, xmin, xmax, ymin, ymax, outname=None):
    if outname == None:
        outname = os.path.splitext(filename)[0] + '_crop' + \
                   os.path.splitext(filename)[1]
    width = xmax - xmin
    height = ymax - ymin
    stream = ffmpeg.input(filename)
    stream = ffmpeg.crop(stream, xmin, ymin, width, height)
    stream = ffmpeg.output(stream, outname)
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, quiet=True)

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

    def _setup_process(self):
        self.process = (
            ffmpeg
            .input(self.filename)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
            )

    def _get_info(self):
        probe = ffmpeg.probe(self.filename)
        video_info = next(
            s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.num_frames = int(video_info['nb_frames'])

class WriteVideoFFMPEG:

    def __init__(self, filename):
        self.filename = filename
        self.frame_no = 0

    def add_frame(self, frame):
        if self.frame_no == 0:
            width = np.shape(frame)[1]
            height = np.shape(frame)[0]
            self._setup_process(width, height)
        self.process.stdin.write(frame.astype(np.uint8).tobytes())
        self.frame_no += 1

    def _setup_process(self, width, height):
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(self.filename, pix_fmt='yuv420p', vcodec='h264', preset='superfast')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )
    def close(self):
        self.process.stdin.close()
        self.process.wait()

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

        #codec_list = ['XVID','mp4v']
        codec_code = list(codec)


        #Check inputs for errors
        if (frame_size is None) and (frame is None):
            raise ArgumentsMissing(('frame_size','frame'))
        if (frame_size is not None) and (frame is not None):
            if frame_size != np.shape(frame):
                raise ImageShapeError(frame,frame_size)

        #Optional arguments to get frame_size and set codec type
        if frame is None:
            self.frame_size = frame_size
        elif frame_size is None:
            frame_size = np.shape(frame)
            self.frame_size = frame_size



        fourcc=cv2.VideoWriter_fourcc(
                                      codec_code[0], codec_code[1],
                                      codec_code[2], codec_code[3]
                                        )


        if filename is None:
            filename = fd.save_filename(caption='select movie filename', file_filter='*.avi;;*.mp4')

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
        
    def add_frame(self, img):
        if self.frame_size == np.shape(img):
            if np.size(self.frame_size) == 2:
                img=cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)

            self.write_vid.write(img)
        else:
            raise ImageShapeError(img, self.frame_size)

    def close(self):
        self.write_vid.release()
        #print('Video closed for writing')

        
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
    
    def __init__(self,filename=None):
        '''
        Initialise video reading object
        if filename = None user must select filename with dialogue
        '''
        if filename == None:
            filename = fd.load_filename(caption='select movie filename', file_filter='*.avi;;*.mp4;;*.*')
        self.filename = filename
        self.open_video()
        self.get_vid_props()
        
    def open_video(self):
        '''Creates a video Object for reading'''
        self.frame_num = 0
        self.read_vid = cv2.VideoCapture(self.filename)

    def get_vid_props(self,show = False):
        self.frame_num = self.read_vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.num_frames = int(self.read_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_time = self.read_vid.get(cv2.CAP_PROP_POS_MSEC)
        self.width = int(self.read_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.read_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.read_vid.get(cv2.CAP_PROP_FPS)
        self.format = self.read_vid.get(cv2.CAP_PROP_FORMAT)
        self.codec = self.read_vid.get(cv2.CAP_PROP_FOURCC)
        self.file_extension = self.filename.split('.')[1]
        
        if show:
            print('----------------------------')
            print('List of Video Properties')
            print('----------------------------')
            print('frame_num : ',self.frame_num)
            print('num_frames : ',self.num_frames)
            print('current_time (ms) : ',self.current_time)
            print('width : ',self.width)
            print('height : ',self.height)
            print('fps : ',self.fps)
            print('format : ',self.format)
            print('codec : ',self.codec)
            print('file_extension :', self.file_extension)
            print('')
            print('unsupported features return 0')
            print('-----------------------------')
                  
    def read_next_frame(self):
        '''reads the next available frame'''
        ret,img = self.read_vid.read() 
        self.frame_num = self.frame_num + 1      
        if ret:
            return img
        else:
            print('Error reading the frame. Check path and filename carefully')

    def find_frame(self,frame_num):
        '''searches for specific frame and reads it'''
        self.set_frame(frame_num)
        self.frame_num = frame_num
        img=self.read_next_frame()
        return img

    def set_frame(self, frame_num):
        """Moves the video reader to the given frame"""
        self.read_vid.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
        self.frame_num = frame_num
        
    def generate_frame_filename(self,ext='.png'):
        len_file_ext = len(self.file_extension)+1
        filename_string = (
                           self.filename[:-len_file_ext]+'_' + 
                           '0'*(len(str(int(self.num_frames))) - 
                           len(str(int(self.frame_num)))) + 
                           str(int(self.frame_num)) + ext
                           )
        return filename_string
        
    def close(self):
        self.read_vid.release()
        #print('Video closed for reading')
        

class ImageShapeError(Exception):
    '''
    Tries to helpfully work out what is wrong with supplied image
    It distinguishes between the image being empty and the image being the 
    wrong way round. (width,height) rather than (height,width) as required
    
    Arguments:
    img is the image being used
    frame_size was the specified dimensions in the instance definition
    '''
    def __init__(self,img,frame_size):
        img_shape = np.shape(img)
        print('Image supplied ',img_shape)
        print('frame_size ', frame_size)
        if img_shape == (frame_size[1],frame_size[0],frame_size[2]):
            print('frame_size should be (h,w,d) not (w,h,d)')
        if img_shape[2] != frame_size[2]:
            print('check colour depth of image')

class ArgumentsMissing(Exception):
    '''
    Designed to inform user they are missing necessary arguments to instantiate
    a class
    '''
    def __init__(self,arguments_missing):
        print('You have failed to supply all necessary arguments')
        print('In this case: ', arguments_missing)


def get_video_dimensions(filename):
    command_w = ['ffprobe',
                '-v', 'error',
                '-show_entries',
                'stream=width',
                '-of',
                'csv=p=0:s=x',
                filename]
    command_h = ['ffprobe',
                 '-v', 'error',
                 '-show_entries',
                 'stream=height',
                 '-of',
                 'csv=p=0:s=x',
                 filename]
    width_out = subprocess.check_output(command_w)
    width = int(width_out.decode("utf-8"))
    height_out = subprocess.check_output(command_h)
    height = int(height_out.decode("utf-8"))
    return width, height

def resize_video(filename, scale=0.5):
    width, height = get_video_dimensions(filename)
    new_width = round(width * scale / 2) * 2
    new_height = round(height * scale / 2) * 2

    command = ['ffmpeg',
               '-i',
               filename,
               '-vf',
               'scale={}:{}'.format(new_width, new_height),
               'out.mp4']
    subprocess.call(command)

# def crop_video(filename, x_min, x_max, y_min, y_max):
#     command = ['ffmpeg -i {} -filter:v "crop={}:{}:{}:{}" out.mp4'.format(
#         filename, x_max-x_min, y_max-y_min, x_min, y_min)]
#     subprocess.call(command)

if __name__ == '__main__':
    # #Create video objects
    # read_vid = ReadVideo()
    # img = read_vid.read_next_frame()
    # write_vid = WriteVideo(frame=img)
    #
    # #Start at frame 100
    # img=read_vid.find_frame(100)
    # write_vid.add_frame(img)
    #
    # #Then add every subsequent frame
    # for frame_num in range(100,read_vid.num_frames-1):
    #     img = read_vid.read_next_frame()
    #     write_vid.add_frame(img)
    #
    # #release resources
    # read_vid.close()
    # write_vid.close()
    in_file = "/home/ppxjd3/Videos/short.mp4"
    out_file = "/home/ppxjd3/Videos/short_ffmpeg.mp4"
    import time
    s = time.time()
    input = ReadVideoFFMPEG(in_file)
    output = WriteVideoFFMPEG(out_file)
    for f in range(input.num_frames):
        frame = input.read_frame()
        frame = ~frame
        output.add_frame(frame)
    output.close()
    print(time.time() - s)
    # out_file = "/home/ppxjd3/Videos/short_opencv.mp4"
    # s2 = time.time()
    # input = ReadVideo(in_file)
    # for f in range(input.num_frames):
    #     frame = input.read_next_frame()
    #     frame = ~frame
    #     if f == 0:
    #         output = WriteVideo(out_file, frame=frame)
    #     else:
    #         output.add_frame(frame)
    # input.close()
    # output.close()
    # print(time.time() - s2)
    
        
        
        
        
        
        
