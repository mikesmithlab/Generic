import numpy as np
import cv2
from tkinter import filedialog
import os


class WriteVideo:
    """
    
    Class is designed to make writing to video files easy
    
    Keyword Arguments:
    filename - Specifies output file. 
                    If filename == True then a dialog will ask for 
                    output file to be supplied
    framesize - Tuple (w,h,colour depth).
                    must match the dimension of the images to be 
                      written
    fps - Framerate for the video
    
    addFrame() - requires an image which matches dimensions of 
                    framesize
    
    close() - releases video object 
                                                                            
    """

    def __init__(self,filename=True,frame_size=(640,480,1),fps=30.0):
        if filename == True:
            filename = filedialog.asksaveasfilename(
                                            defaultextension='.mp4',
                                            filetypes = [('MP4','.mp4')]
                                            )
        fourcc=cv2.VideoWriter_fourcc('X','V','I','D')
        self.write_vid = cv2.VideoWriter(
                                        filename,
                                        fourcc,
                                        fps,
                                        (frame_size[0],frame_size[1]),
                                        frame_size[2]
                                        )
        print('Video open for writing')
    
    def add_frame(self,img):
        '''img dimensions must match framesize'''
        self.write_vid.write(img)
    
    def close(self):
        self.write_vid.release()
        print('Video closed for writing')

        
class ReadVideo:
    '''
    
    Class is designed to handle the reading of videos and 
    extraction of frames
        
    Keyword Arguments:
    filename - Specifies output file. 
                    If filename == True then a dialog will ask for 
                    output file to be supplied
        
    Variables:
        self.filename - filename
        self.framenum - current frame
        self.read_vid - Video Object
        self.num_frames - Number of Frames in Video
        self.width - width of frame
        self.height - height of frame
        self.fps - Video framerate
        self.current_time - Current time in ms 
                            (Not always supported)
        self.format - format of video (Not always supported)
        self.codec - codec of video
    
    Methods:
        __init__(filename=True) - Create Video Object
        open_video() - opens Video for reading
        close_video() - closes Video for reading
        read_next_frame() - read the next frame in video
        find_frame(frame_num) - find frame specified by framenum
        export_section_vid(filename=True,frames=[]) - export 
                            specified frames to new Video. 
                            frames can be numpy array or list
        get_vid_props(show=True) - get video properties and print 
                           to terminal if show==True
        
        
    '''
    def __init__(self,filename=True):
        self.open_video(filename=filename)
        
    def open_video(self,filename=True):
        '''Creates a video Object for reading'''
        try:
            vid_open = self.read_vid.isOpened()
        except:
            vid_open = False
        
        if vid_open == False:
            if filename == True:
                filename = filedialog.askopenfilename(defaultextension='.mp4',filetypes = [('MP4','.mp4')])
            self.filename=filename
            self.frame_num = 0
            self.read_vid= cv2.VideoCapture(filename)
    
    def close_video(self):
        self.read_vid.release()
                
    def read_next_frame(self):
        '''reads the next available frame'''
        try:
            ret,img = self.read_vid.read() 
            self.frame_num = self.frame_num + 1
        except:
            pass
        if ret:
            return img
        else:
            print('Error reading the frame. Check path and filename carefully')

    def find_frame(self,frame_num):
        '''searches for framenum and reads this value.'''
        self.read_vid.set(cv2.CAP_PROP_POS_FRAMES,float(frame_num))
        self.frame_num = frame_num
        img=self.read_next_frame()
        return img
    
    def export_section_vid(self,new_file = False,frames=range(10)):
        '''
        Cut selection of frames from Video and save to new file.
        
        new_file - optional name for new file. Defaults to original file 
                    name with '_export' added
        frames - is a list or numpy array of frame numbers
        '''
        if type(frames) == type([1,2]):
            frames = np.array(frames)
            
        if new_file == False:
            filenames = os.path.splitext(self.filename)
            new_file = filenames[0] + '_export' + filenames[1]
        #Perform checks for consistency
        self.get_vid_props()
        if np.max(frames) >= self.num_frames:
            print('Frame selection exceeds length of original video')
        
        self.op_vid = WriteVideo(filename = new_file)
        
        
        

    def get_vid_props(self,show = False):
        self.frame_num = self.read_vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.num_frames = self.read_vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.current_time = self.read_vid.get(cv2.CAP_PROP_POS_MSEC)
        self.width = self.read_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.read_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.read_vid.get(cv2.CAP_PROP_FPS)
        self.format = self.read_vid.get(cv2.CAP_PROP_FORMAT)
        self.codec = self.read_vid.get(cv2.CAP_PROP_FOURCC)
        
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
            print('')
            print('unsupported features return 0')
            print('-----------------------------')

if __name__ == '__main__':
        write_vid = WriteVideo()
        vid = ReadVideo()
        vid.get_vid_props()
        
        
        