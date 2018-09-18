import glob
from os.path import split

def get_files_directory(path,full_filenames=True):
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
    if full_filenames==True:
        return filename_list
    else:
        f = [split(f)[1] for f in filename_list]
        return f