from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QDialog
import sys
import os
import glob

def load_filename(caption='Find a filename',
                 directory='/home/ppxjd3/Code/Generic/',
                 file_filter='*.png;;*.jpg;;*.*'):
    app = QApplication(sys.argv)
    filename = QFileDialog.getOpenFileName(parent=None,
                                           caption=caption,
                                           directory=directory,
                                           filter=file_filter)[0]
    app.exit()
    return filename





def get_files_directory(path, full_filenames=True):
    '''
    Inputs:
    Given a path it will return a list of all files as a list
    path can include wild cards see example
    full_filenames =True joins filenames to path

    Returns:
    List of filenames with or without paths

    Example:
    file_list = get_files_directory('~/*.png')


    '''
    filename_list = glob.glob(path)
    if full_filenames == True:
        return filename_list
    else:
        f = [os.path.split(f)[1] for f in filename_list]
        return f

def save_filename(caption='Save File',
              directory='/home/ppxjd3/Code/Generic/',
              file_filter='*.mp4;;*.avi'):
    app = QApplication(sys.argv)
    output = QFileDialog.getSaveFileName(parent=None,
                                           caption=caption,
                                           directory=directory,
                                           filter=file_filter)
    file, extension = os.path.splitext(output[0])
    if extension != "":
        filename = output[0]
    else:
        filename = file + output[1][1:]
    return filename


if __name__ == "__main__":
    file = load_filename()
    print('file = ', file)

    new_file = save_filename()
    print('save_file = ', new_file)

