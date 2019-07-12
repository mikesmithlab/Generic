from moviepy.editor import AudioFileClip

"""Functions for extracting the frequencies of square waves in a 
sound file."""

def extract_wav(file):
    audioclip = AudioFileClip(file)
    audioclip_arr = audioclip.to_soundarray(fps=48000, nbytes=2)
    return audioclip_arr

