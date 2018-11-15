import cv2
import tkinter
from tkinter import filedialog
import time
from Generic.video import WriteVideo
import os

'''
If you are going to add a camera to this list please test and fill in all
available settings.
'''
camera_settings = {
              'logitechHD1080p':{
                                 'res':((1920,1080,3),(640,480,3),(1280,720,3),(480,360,3)),
                                 'fps':((30.0),)
                                 },
              'philips 3':{
                           'res':((640,480,3),(1280,1080,3)),
                           'fps':((20.0),)
                          }, 
              'mikelaptop':{
                           'res':((640,480,3)),
                           'fps':((20.0),)
                           }
              }


class Camera:
    def __init__(self,cam_type = 'logitechHD1080p',cam_num=0,frame_size=-1,fps=-1):
        '''
        This is a class that handles all the common actions
        
        camtype specifies type of camera:
            - 'logitechHD1080p'
            - 'philips 3 '
            
        camnum specifies which camera. Sometimes needs setting
        
        framesize is a tuple (W,H,Color Depth) or if -1 specifies default
        '''
        self.cam_type = cam_type   
        self.cam = cv2.VideoCapture(cam_num)
        
        '''All camera frame sizes and frame rates are defined in CameraDetailsList'''
        self.frame_sizes = camera_settings[cam_type]['res']
        self.frame_rates = camera_settings[cam_type]['fps']

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
        print(self.frame_size)
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.frame_size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.frame_size[1]) 
        self.cam.set(cv2.CAP_PROP_FPS,self.fps)
        
        ret,frame = self.cam.read()
        if ret:
            print('Camera working')
        else:
            raise FrameReadingError()
             

    def preview(self):
        '''This produces a live preview window to enable you to optimise camera settings
        If you have more than one camera you need to change the number of the camera which is 
        linked to the usb port being us ed
        press q to quit the preview'''
    
        time.sleep(2)
        loopvar=True
        while(loopvar==True):
            # Capture frame-by-frame
            ret, frame = self.cam.read()
            
            if not ret:
                print('No frame returned from camera. Often this is a frame size issue')
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cap.release()
                cv2.destroyAllWindows()
                loopvar = False
                
    def write_to_vid(self,filename=None,fps=30.0):
        if filename == None:
            filename = filedialog.asksaveasfilename(
                                            defaultextension=extensions[0][1],
                                            filetypes = extensions
                                            )
        self.vid = WriteVideo(filename=filename,frame_size=(self.frame_size[1],self.frame_size[0],self.frame_size[2]),fps=self.fps)
        
        
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

    def single_pic_img(self,filename=None,show_pic=False):
        if filename == None:
            filename = filedialog.asksaveasfilename(
                                            defaultextension=extensions[0][1],
                                            filetypes = extensions
                                            )
        '''writes single image to picture file '''
        ret,frame=self.cam.read()        
        cv2.imwrite(filename,frame)
        if show_pic:
                cv2.imshow('frame',frame)

    def single_pic_array(self):
        """Return single image as array"""
        ret, frame = self.cam.read()
        return frame
    
    
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
        self.auto_exposure = self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        
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
            print('auto_exposure:', self.auto_exposure)
            print('')
            print('unsupported features return 0')
            print('-----------------------------')
    
    def set_cam_props(self, property_name, value):
        '''
        Legitimate properties are brightness,contrast,hue,saturation,gain
        some cameras will not support all of them.
        
        fps can be set for record.
        
        Frame size require you to initialise a new camera object.
        First call object.close() then reinitialise obj = Camera(framesize = -1)

        Set auto exposure to 0.25 for manual control
        '''
        property_names = (
                'brightness', 'contrast', 'gain',
                'saturation', 'hue', 'exposure',
                'auto exposure')
        cv_property_codes = (
                cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST,
                cv2.CAP_PROP_GAIN, cv2.CAP_PROP_SATURATION,
                cv2.CAP_PROP_HUE, cv2.CAP_PROP_EXPOSURE,
                cv2.CAP_PROP_AUTO_EXPOSURE)
        if property_name in property_names:
            property_code = cv_property_codes[property_names.index(property_name)]
            try:
                self.cam.set(property_code, value)
            except:
                raise CamPropsError(property_name, True)
        else:
            raise CamPropsError(property_name, False)
        self.get_cam_props()
        
    def save_cam_settings(self,filename=None):
        if filename == None:
            filename = filedialog.asksaveasfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])
        
        self.get_cam_props(show=True)
        settings = (self.brightness,self.contrast,self.gain,self.saturation,self.hue,self.exposure)
        with open(filename,'w') as settings_file:
            for item in settings:
                settings_file.write("%s\n" % item)
               
    def load_cam_settings(self,filename=None):
        if filename == None:
            filename = filedialog.askopenfilename(defaultextension='.camlog',filetypes = [('CAM','.camsettingslog')])        
            
        with open(filename,'r') as settings_file:
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
      
    def search(self,cam_type):
        try:
            return self.camera_list[cam_type]
        except:
            print('Camera Details List Error')

def find_camera_number():
    items = os.listdir('/dev/')
    newlist = []
    for names in items:
        if names.startswith("video"):
            newlist.append(names)
    return int(newlist[0][5:])

class CamPropsError(Exception):
    def __init__(self,property_name,property_exists):
        if property_exists == True:
            print('Error setting camera property')
        else:
            print(property_name, 'does not exist')

class FrameReadingError(Exception)  :
    def __init__(self):
        print('Error reading the frame, often this is due to incorrect frame shape')
        print('check you have height and width the correct way round')
    
        
if __name__ == '__main__':
    web_cam = Camera(cam_type='philips 3')
    web_cam.preview()
    web_cam.write_to_vid('test.mp4')
    web_cam.start_record(num_frames=15,show_pic=True,time_lapse=1)
    web_cam.close_write_vid()