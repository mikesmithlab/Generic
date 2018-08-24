import cv2
import tkinter
from tkinter import filedialog
import time
from Video import WriteVideo
from camconfig import*



class Camera:
    def __init__(self,camtype = 'logitechHD1080p',camnum=0,framesize = -1,fps=-1):
        '''
        initialisation creates the camera object.
        
        camtype specifies type of camera:
            - 'logitechHD1080p'
            - 'philips 3 '
            
        camnum specifies which camera. Sometimes needs setting
        
        framesize is a tuple (W,H,Color Depth) or if -1 specifies default
        '''
        self.camtype = camtype   
        #Create camera object
        self.cam = cv2.VideoCapture(camnum)
        
        '''------------------------------------------------------------------
                List all camera types here with params
                If adding a camera please test thoroughly
        ---------------------------------------------------------------------
        '''
        if camtype == 'logitechHD1080p':
            self.frame_sizes,self.framerates = readLogitech()
            #Format (W,H,colourdepth). Default value 1st
            #framerates. Default value 1st
        elif camtype == 'philips 3':
            self.frame_sizes,self.framerates = readPhilips()
        else:
            print('camera not supported. Update camconfig.py with info')
        
        '''------------------------------------------------------------------
        ------------------------------------------------------------------'''
        self.default_frame_size = self.frame_sizes[0]
        self.defaultfps = self.framerates[0]
       
        #Set framerate
        if fps == -1:
            self.fps = self.defaultfps
        elif fps in self.framerates:
            self.fps = fps
        else:
            print('fps not possible')
        
        
        #Set framesize
        if framesize == -1:
            self.framesize = self.default_frame_size
        elif framesize in self.frame_sizes:
            self.framesize = framesize
        else:
            print('frame size not possible')           
            
        #Set resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.framesize[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.framesize[1]) 
        self.cam.set(cv2.CAP_PROP_FPS,self.fps)
        
        try:
            ret,frame = self.cam.read()
            print('Camera working')
        except:
            print('error reading test frame')
             

    def previewCam(self):
        '''This produces a live preview window to enable you to optimise camera settings
        If you have more than one camera you need to change the number of the camera which is 
        linked to the usb port being used
        press q to quit the preview'''
    
        time.sleep(2)
        loopvar=True
        while(loopvar==True):
            # Capture frame-by-frame
            ret, frame = self.cam.read()
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cap.release()
                cv2.destroyAllWindows()
                loopvar = False
                
    def createWriteVid(self,filename,fps=30.0):
        try:
            self.vid.close()
        except:
            pass
        self.vid = WriteVideo(filename,framesize=self.framesize,fps=self.fps)
        
        
    def closeWriteVid(self):
        self.vid.close()

    def startRecord(self,numframes = -1,fps=-1,showPic=False,timelapse=-1):
        '''
        To avoid delay in recording you must call createWriteVid prior to startRecord
        starts recording video for numframes. If numframes set to -1 the recording is
        continuous until q is pressed on keyboard
        fps will throw error if the fps is not listed under the list of framerates in __init__
        if fps = -1 it will use the default fps which is the first value listed in self.framerates
        timelapse adds a delay in seconds between collecting images unless it = -1 in 
        which case no delay is added.
        '''
        
        
        
        n = 0     
        loopvar = True
        while(loopvar==True):
            # Capture frame from camera
            ret, frame = self.cam.read()
            #write frame to video
            self.vid.addFrame(frame)
            # Display the resulting frame
            if showPic:
                cv2.imshow('frame',frame)
            #stop when q is pressed
            
            if timelapse != -1:
                time.sleep(timelapse)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.vid.close()
                cv2.destroyAllWindows()
                loopvar = False
            n=n+1
            #Stop after numframes
            if n == numframes-1:
                loopvar = False
                self.vid.close()
                cv2.destroyAllWindows()
            
            
                

    def singlePicImg(self,filename,showPic=False):
        '''writes single image to picture file '''
        ret,frame=self.cam.read()        
        cv2.imwrite(filename,frame)
        if showPic:
                cv2.imshow('frame',frame)
    
    
    def singlePicVid(self,showPic=False):
        '''Stores single picture from camera to video
        Before calling you must call createWriteVid
        When you have finished collecting video call closeWriteVid '''
        ret, frame = self.cam.read()
        #write frame to video
        self.vid.addFrame(frame) 
        if showPic:
                cv2.imshow('frame',frame)
    
    def getCamProps(self,show=False):
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.format = self.cam.get(cv2.CAP_PROP_FORMAT)
        self.mode = self.cam.get(cv2.CAP_PROP_MODE)
        self.saturation = self.cam.get(cv2.CAP_PROP_SATURATION)
        self.gain = self.cam.get(cv2.CAP_PROP_GAIN)
        self.hue = self.cam.get(cv2.CAP_PROP_HUE)
        self.contrast = self.cam.get(cv2.CAP_PROP_CONTRAST)
        self.brightness = self.cam.get(cv2.CAP_PROP_BRIGHTNESS)
        self.exposure = self.cam.get(cv2.CAP_PROP_EXPOSURE)
        
        if show:
            print('----------------------------')
            print('List of Video Properties')
            print('----------------------------')
            print('width : ',self.width)
            print('height : ',self.height)
            print('fps : ',self.fps)
            print('format : ',self.format)
            print('mode : ',self.mode)
            print('brightness : ',self.brightness)
            print('contrast : ',self.contrast)
            print('hue : ',self.hue)
            print('saturation : ',self.saturation)
            print('gain : ',self.gain)
            print('exposure :',self.exposure)
            print('')
            print('unsupported features return 0')
            print('-----------------------------')
    
    def setCamProps(self,property_name,value):
        '''
        Legitimate properties are brightness,contrast,hue,saturation,gain
        some cameras will not support all of them.
        
        fps can be set for record.
        
        Frame size require you to initialise a new camera object.
        First call object.close() then reinitialise obj = Camera(framesize = -1)
        '''
        properties_list = ['brightness','contrast','gain','saturation','hue','exposure']
        cv_property_code = [cv2.CAP_PROP_BRIGHTNESS,cv2.CAP_PROP_CONTRAST,cv2.CAP_PROP_GAIN,
                            cv2.CAP_PROP_SATURATION,cv2.CAP_PROP_HUE,cv2.CAP_PROP_EXPOSURE]
        if property_name in properties_list:
            property_code = cv_property_code[properties_list.index(property_name)]
            try:
                self.cam.set(property_code,value)
            except:
                print('Error setting property')
        else:
            print('Error property not in list')
        self.getCamProps()
        
    def saveCamSettings(self):
        camSettingsFile = filedialog.asksaveasfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])
        
        self.getCamProps(show=True)
        settings = [self.brightness,self.contrast,self.gain,self.saturation,self.hue,self.exposure]
        with open(camSettingsFile,'w') as settingsFile:
            for item in settings:
                settingsFile.write("%s\n" % item)
        
        
    def loadCamSettings(self):
        camSettingsFile = filedialog.askopenfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])        
        with open(camSettingsFile,'r') as settingsFile:
            settingsList = settingsFile.read().splitlines() 
        self.brightness = settingsList[0]
        self.contrast = settingsList[1]
        self.gain = settingsList[2]
        self.saturation = settingsList[3]
        self.hue = settingsList[4]
        self.exposure = settingsList[5]
                    
        self.setCamProps('brightness',self.brightness)
        self.setCamProps('contrast',self.contrast)
        self.setCamProps('gain',self.gain)
        self.setCamProps('saturation',self.saturation)
        self.setCamProps('hue',self.hue)
        self.setCamProps('exposure',self.exposure)
            

    def close(self):
        self.cam.release()
        
        print('Camera closed')
        
if __name__ == '__main__':
    '''clean up code in case the camera closed with an error'''
    try:
        capture.close()
    except:
        pass
    
    
    root = tkinter.Tk()
    root.withdraw()
    
    '''Simple dialogue to get filename to save images to.'''
    #filename = filedialog.asksaveasfilename(defaultextension='.avi',filetypes = [('AVI','.avi')])
    #filename2 = filedialog.asksaveasfilename(defaultextension='.png',filetypes = [('PNG','.png')])
    filename3 = filedialog.asksaveasfilename(defaultextension='.mp4',filetypes = [('MP4','.mp4')])
    root.destroy() 

    '''Create a camera object'''
    capture = Camera(framesize = (1920,1080,3))
    
    '''Preview it'''
    capture.previewCam()  
    
    #capture.getCamProps(show=True)
    #capture.saveCamSettings()
    #capture.setCamProps('exposure',0.1)
    #capture.loadCamSettings()
    
    '''write a single png '''
    #capture.singlePicImg(filename2)
    
    '''record a video with numframes. Vid can be stopped by hitting q'''    
    capture.createWriteVid(filename=filename3)
    capture.startRecord(numframes = 100,showPic=True,timelapse=1)
    
    #Clean up
    capture.close()    
    