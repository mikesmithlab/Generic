'''
Camera configuration file

'''

def readLogitech():
    'logitechHD1080p'
    frame_sizes = [(1920,1080,3),(640,480,3),(1280,720,3),(480,360,3)]
    #List all the possible framerates with default 1st in list
    framerates = [30.0]
    return (frame_sizes,framerates)

def readPhilips():
    'philips 3'
    frame_sizes = [(640,480,3),(1280,1080,3)]
    framerates = [20.0]
    return (frame_sizes,framerates)

def readNewCamera():
    '''
    Format example
    '''
    frame_sizes = []
    framerates = []
    return (frame_sizes,framerates)