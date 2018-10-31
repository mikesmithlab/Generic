from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QDialog
import sys


def get_filename(caption='Find a filename',
                 directory='/home/ppxjd3/Code/Generic/',
                 file_filter='*.png'):
    app = QApplication(sys.argv)
    filename = QFileDialog.getOpenFileName(parent=None,
                                           caption=caption,
                                           directory=directory,
                                           filter=file_filter)[0]
    app.exit()
    return filename


def save_file(caption='Save File',
              directory='/home/ppxjd3/Code/Generic/',
              file_filter='*.mp4'):
    app = QApplication(sys.argv)
    filename = QFileDialog.getSaveFileName(parent=None,
                                           caption=caption,
                                           directory=directory,
                                           filter=file_filter)[0]
    return filename


if __name__ == "__main__":
    file = get_filename()
    print('file = ', file)

    new_file = save_file()
    print('save_file = ', new_file)

