# Generic
This is a repository for all the highly generic code that will be reused in project after project. It consists of the following modules:

-audio
    - Functions for extracting the frequencies of square waves in a 
sound file.
- camera
    - Allows taking images and videos with the different cameras in the lab with generic user interface
- filedialogs
    - Functions and classes for using file dialogs
- fitting
    - Allows simple fitting of various functional forms together with viewing data and selecting appropriate values. Also provides simple statistics.
- images
    - Contains methods for manipulating images
- plotting
    - Controls plotting in matplotlib. Mainly for multipanel figures.
    - Also enables easy plot of histogram
- signal_toolbox
    - Fourier transforms
- video
    - Allows reading writing videos


Also contains a number of arduino sketches:
- CameraTrigger.ino
    - Periodically triggers a camera to record