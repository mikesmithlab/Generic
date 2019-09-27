import cv2
from Generic.filedialogs import save_filename, load_filename
import time
import Generic.video as video
import os
import sys
import subprocess
import signal
import datetime
from sh import gphoto2
from Generic.images.basics import display
from Generic.filedialogs import get_files_directory


class GP2Camera:
    def __init__(self, cam_type='Nikon Coolpix S9700', transfer_folder_name='/home/mike/Pictures/', frame_size=-1,
                 fps=-1):
        self.folder_name=transfer_folder_name
        self.kill_process()
        #time.sleep(3)
        self.setup_folders()
        self.file_list = get_files_directory(self.download_folder + '/*.JPG')

    def kill_process(self):
        #kill the gphoto2 process at power on
        p = subprocess.Popen(['ps','-A'],stdout=subprocess.PIPE)
        out, err = p.communicate()

        for line in out.splitlines():
            if b'gvfsd-gphoto2' in line:
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

    def timestamp(self):
        shot_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return shot_time

    def setup_folders(self):
        datetimeval = self.timestamp()
        self.download_folder = self.folder_name + datetimeval
        try:
            os.mkdir(self.download_folder)
        except:
            print('Warning: folder already exists')
        os.chdir(self.download_folder)

    def trigger(self):
        gphoto2('--capture-image-and-download')
        self.renameFile()

    def preview(self):
        gphoto2('--capture-image-and-download')
        self.renameFile(name='preview')
        preview_img = cv2.imread(self.download_folder + '/preview.jpg')
        display(preview_img)
        os.remove(self.download_folder + '/preview.jpg')

    def capture_time_lapse(self, num_imgs=1, interval=6):
        min_interval = 6
        if interval < min_interval:
            print('Minimum interval = ', min_interval)
        for i in range(num_imgs):
            self.trigger()
            time.sleep(interval)

    def renameFile(self, name=None):
        file_list = get_files_directory(self.download_folder + '/*.JPG')
        for file in file_list:
            if file not in self.file_list:
                if name is None:
                    datetimeval=self.timestamp()
                    os.rename(file, self.download_folder + '/' + datetimeval + '.jpg')
                    self.file_list.append(self.download_folder + datetimeval + '.jpg')
                else:
                    os.rename(file, self.download_folder + '/preview.jpg')

    def camera_settings(self, settings_script):
        time.sleep(1)
        p = subprocess.Popen([settings_script], stdout=subprocess.PIPE)
        time.sleep(2)

















if __name__ == '__main__':
    gp = GP2Camera()

    gp.camera_settings('/home/mike/PycharmProjects/Generic/gphoto2_cam_config')

    gp.preview()
    #gp.trigger()
   
    #gp.capture_time_lapse(num_imgs=10, interval=10)
