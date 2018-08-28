import cv2
import tkinter
from tkinter import filedialog
import time
from Video import WriteVideo
from camconfig import*





class Camera:
    def __init__(self,cam_type = 'logitechHD1080p',cam_num=0,frame_size = -1,fps=-1):
        '''
        initialisation creates the camera object.
        
        camtype specifies type of camera:
            - 'logitechHD1080p'
            - 'philips 3 '
            
        camnum specifies which camera. Sometimes needs setting
        
        framesize is a tuple (W,H,Color Depth) or if -1 specifies default
        '''
        self.cam_type = cam_type   
        #Create camera object
        self.cam = cv2.VideoCapture(cam_num)
        
        '''All camera frame sizes and frame rates are defined in CameraDetailsList'''
        cam_details = CameraDetailsList()
        self.frame_sizes, self.frame_rates = cam_details.search(cam_type)

        self.default_frame_size = self.frame_sizes[0]
        self.default_fps = self.frame_rates[0]
       
        #Set framerate
        if fps == -1:
            self.fps = self.default_fps
        elif fps in self.frame_rates:
            self.fps = fps
        else:
            print('fps not possible')
        
        
        #Set framesize
        if frame_size == -1:
            self.frame_size = self.default_frame_size
        elif frame_size in self.frame_sizes:
            self.frame_size = frame_size
        else:
            print('frame size not possible')           
            
        #Set resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.frame_size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.frame_size[1]) 
        self.cam.set(cv2.CAP_PROP_FPS,self.fps)
        
        try:
            ret,frame = self.cam.read()
            print('Camera working')
        except:
            print('error reading test frame')
             

    def preview_cam(self):
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
                
    def create_write_vid(self,filename,fps=30.0):
        try:
            self.vid.close()
        except:
            pass
        print(self.frame_size)
        self.vid = WriteVideo(filename=filename,frame_size=self.frame_size,fps=self.fps)
        
        
    def close_write_vid(self):
        self.vid.close()

    def start_record(self,num_frames = -1,fps=-1,show_pic=False,time_lapse=-1):
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
            self.vid.add_frame(frame)
            # Display the resulting frame
            if show_pic:
                cv2.imshow('frame',frame)
            #stop when q is pressed
            
            if time_lapse != -1:
                time.sleep(time_lapse)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.vid.close()
                cv2.destroyAllWindows()
                loopvar = False
            n=n+1
            #Stop after numframes
            if n == num_frames-1:
                loopvar = False
                self.vid.close()
                cv2.destroyAllWindows()
            
            
                

    def single_pic_img(self,filename,show_pic=False):
        '''writes single image to picture file '''
        ret,frame=self.cam.read()        
        cv2.imwrite(filename,frame)
        if show_pic:
                cv2.imshow('frame',frame)
    
    
    def single_pic_vid(self,show_pic=False):
        '''Stores single picture from camera to video
        Before calling you must call createWriteVid
        When you have finished collecting video call closeWriteVid '''
        ret, frame = self.cam.read()
        #write frame to video
        self.vid.add_frame(frame) 
        if show_pic:
                cv2.imshow('frame',frame)
    
    def get_cam_props(self,show=False):
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
    
    def set_cam_props(self,property_name,value):
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
        self.get_cam_props()
        
    def save_cam_settings(self):
        cam_settings_file = filedialog.asksaveasfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])
        
        self.get_cam_props(show=True)
        settings = [self.brightness,self.contrast,self.gain,self.saturation,self.hue,self.exposure]
        with open(cam_settings_file,'w') as settings_file:
            for item in settings:
                settings_file.write("%s\n" % item)
        
        
    def load_cam_settings(self):
        cam_settings_file = filedialog.askopenfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])        
        with open(cam_settings_file,'r') as settings_file:
            settings_list = settings_file.read().splitlines() 
        self.brightness = settings_list[0]
        self.contrast = settings_list[1]
        self.gain = settings_list[2]
        self.saturation = settings_list[3]
        self.hue = settings_list[4]
        self.exposure = settings_list[5]
                    
        self.set_cam_props('brightness',self.brightness)
        self.set_cam_props('contrast',self.contrast)
        self.set_cam_props('gain',self.gain)
        self.set_cam_props('saturation',self.saturation)
        self.set_cam_props('hue',self.hue)
        self.set_cam_props('exposure',self.exposure)
            

    def close(self):
        self.cam.release()
        
        print('Camera closed')

'''
-----------------------------------------------------------------------------
Specific Camera implementations
-----------------------------------------------------------------------------
'''
class CameraDetailsList:
    '''
    CameraDetailsList is a helper class which stores the details of all the 
    different cameras. Mainly this is done to make the code more readable
    and to make it easier to add additional cameras.
    
    To add a new camera give is a name and add a Tuple to the dictionary
    ([List of tuples specifying frame sizes],[list of frame rates])
    
    The default values should be placed first in the list.
    '''
    
    def __init__(self):
        self.camera_list = {'logitechHD1080p':([(1920,1080,3),(640,480,3),(1280,720,3),(480,360,3)],[30.0]),
                   'philips 3':([(640,480,3),(1280,1080,3)],[20.0])
                   }
        
    def search(self,cam_type):
        return self.camera_list[cam_type]
    
   
        
    
        
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
    capture = Camera(cam_type='logitechHD1080p',frame_size = (1920,1080,3))
    
    '''Preview it'''
    capture.preview_cam()  
    
    #capture.get_cam_props(show=True)
    #capture.save_cam_settings()
    #capture.set_cam_props('exposure',0.1)
    #capture.load_cam_settings()
    
    '''write a single png '''
    #capture.single_pic_img(filename2)
    
    '''record a video with numframes. Vid can be stopped by hitting q'''    
    capture.create_write_vid(filename=filename3)
    capture.start_record(num_frames = 100,show_pic=True,time_lapse=1)
    
    #Clean up
    capture.close()    
    