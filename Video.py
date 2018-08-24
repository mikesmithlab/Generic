import numpy as np
import cv2
from tkinter import filedialog
import os

class WriteVideo:
    '''
    Class is designed to make writing to video files easy
    '''
    def __init__(self,filename=True,fps=30.0,framesize=(640,480,1)):
        if filename == True:
            filename = filedialog.askopenfilename(defaultextension='.mp4',filetypes = [('MP4','.mp4')])
        fourcc=cv2.VideoWriter_fourcc('X','V','I','D')
        self.writeVid = cv2.VideoWriter(filename,fourcc,fps,(framesize[0],framesize[1]),framesize[2])
        print('Video open for writing')
    
    def addFrame(self,img):
        self.writeVid.write(img)

    
    def close(self):
        self.writeVid.release()
        print('Video closed for writing')
        
class ReadVideo:
    '''
    Class is designed to handle the reading of videos and extraction of frames
    '''
    def __init__(self,filename=True):
        self.openVideo(filename=filename)
        
    def openVideo(self,filename=True):
        try:
            vidOpen = self.readVid.isOpened()
        except:
            vidOpen = False
        
        if vidOpen == False:
            if filename == True:
                filename = filedialog.askopenfilename(defaultextension='.mp4',filetypes = [('MP4','.mp4')])
            self.filename=filename
            self.framenum = 0
            self.readVid= cv2.VideoCapture(filename)
    
    def closeVideo(self):
        self.readVid.release()
                
    def readNextFrame(self):
        try:
            ret,img = self.readVid.read() 
            self.framenum = self.framenum + 1
        except:
            pass
        if ret:
            return img
        else:
            print('Error reading the frame. Check path and filename carefully')

    def findFrame(self,framenum):
        self.readVid.set(cv2.CAP_PROP_POS_FRAMES,float(framenum))
        self.framenum = framenum
        img=self.readNextFrame()
        return img
    
    def exportSectionVid(self,newfile = False,frames=range(10)):
        '''
        Cut selection of frames from Video and save to new file.
        newfile - optional name for new file. Defaults to original file name with 
        '_export' added
        frames is a list or numpy array of frame numbers
        '''
        if type(frames) == type([1,2]):
            frames = np.array(frames)
            
        if newfile == False:
            filenames = os.path.splitext(self.filename)
            newfile = filenames[0] + '_export' + filenames[1]
        #Perform checks for consistency
        self.getVidProps()
        if np.max(frames) >= self.numframes:
            print('Frame selection exceeds length of original video')
        
        self.op_vid = WriteVideo(filename = newfile)
        
        img = self.findFrame
        

    def getVidProps(self,show = False):
        self.framenum = self.readVid.get(cv2.CAP_PROP_POS_FRAMES)
        self.numframes = self.readVid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.currenttime = self.readVid.get(cv2.CAP_PROP_POS_MSEC)
        self.width = self.readVid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.readVid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.readVid.get(cv2.CAP_PROP_FPS)
        self.format = self.readVid.get(cv2.CAP_PROP_FORMAT)
        self.codec = self.readVid.get(cv2.CAP_PROP_FOURCC)
        
        if show:
            print('----------------------------')
            print('List of Video Properties')
            print('----------------------------')
            print('framenum : ',self.framenum)
            print('numframes : ',self.numframes)
            print('currenttime (ms) : ',self.currenttime)
            print('width : ',self.width)
            print('height : ',self.height)
            print('fps : ',self.fps)
            print('format : ',self.format)
            print('codec : ',self.codec)
            print('')
            print('unsupported features return 0')
            print('-----------------------------')

if __name__ == '__main__':
        vid = ReadVideo()
        vid.getVidProps()
        
        
        