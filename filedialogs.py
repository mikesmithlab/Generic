from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QDialog
import sys
import os
import glob


def save_filename(caption='Save File',
              directory='/home/ppxjd3/Code/Generic/',
              file_filter='*.mp4;;*.avi'):
    """
    Choose a save filename using a dialog.

    Parameters
    ----------
    caption: str
        Title for the window

    directory: str
        Path to open the file dialog at

    file_filter:str
        String containing extension wildcards separated by ;;

    Returns
    -------
    filename: str
        The save filename including path and extension
    """
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


def load_filename(caption='Find a filename',
                 directory='/home/ppxjd3/Code/Generic/',
                 file_filter='*.*;;*.png;;*.jpg',
                 remove_ext=False):
    """
    Choose a load filename using a dialog.

    Parameters
    ----------
    caption: str
        Title for the window

    directory: str
        Path to open the file dialog at

    file_filter:str
        String containing extension wildcards separated by ;;

    Returns
    -------
    filename: str
        The load filename including path and extension
    """
    app = QApplication(sys.argv)
    filename = QFileDialog.getOpenFileName(parent=None,
                                           caption=caption,
                                           directory=directory,
                                           filter=file_filter)[0]
    app.exit()
    if remove_ext:
        filename = os.path.splitext(filename)[0]
    return filename

def get_files_directory(path, full_filenames=True):
    '''
    Gets list of files from a directory

    Given a path it will return a list of all files as a list

    Parameters
    ----------
    path: str
        filepath to load files from, including wildcards will only load
        files which fit the wildcard

    full_filenames: Bool
        If True joins filenames to path

    Returns
    -------
    List of filenames with or without paths

    Example
    -------
    file_list = get_files_directory('~/*.png')
    '''
    filename_list = glob.glob(path)
    if full_filenames == True:
        return filename_list
    else:
        f = [os.path.split(f)[1] for f in filename_list]
        return f


class BatchProcess:
    """
    Generator for batch processing of files in scripts

   The BatchProcess() object if called without a path filter
   such as ~/ppzmis/*ab*.csv will open a dialogue. If you click on a
   file in a folder it will create a list of all the filenames with
   the same type of extension.

   The object can then be iterated over yielding a new filename until
   there are no more left. Easiest way to set up is:

   for filename in BatchProcess():
       load file for processing
       function_of_script(filename)
    """
    def __init__(self,pathfilter=None):
        if pathfilter is None:
            filename = load_filename(caption='Select file in directory')
            path = os.path.split(filename)[0]
            file, extension = os.path.splitext(filename)
            extension='*' + extension
            pathfilter = os.path.join(path, extension)

        self.filenames = get_files_directory(pathfilter)
        self.num_files = len(self.filenames)
        self.current = 0


    def __iter__(self):
        return self

    def __next__(self):
        try:
            filename = self.filenames[self.current]
            self.current += 1
        except IndexError:
            raise StopIteration
        return filename



if __name__ == "__main__":
    file = load_filename()
    print('load file = ', file)

    new_file = save_filename()
    print('save_file = ', new_file)

    #2 possibilities for using BatchProcess()
    for filename in BatchProcess():
       print(filename)
       #call your script with filename, load file whatever
       #Script can take a filefilter e.g '~/ppzmis/*ball*.txt' to filter files in generator

    #use it like a normal generator expression.
    batch = BatchProcess()
    filename = next(batch)
    print(filename)
    filename = next(batch)
    print(filename)