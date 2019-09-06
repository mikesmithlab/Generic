from Generic.filedialogs import BatchProcess
from Generic.video import ReadVideo, WriteVideo


def concatenate_movies_in_folder(pathfilter, output_path, frame_size=(2048,2048,3)):
    for file in BatchProcess(pathfilter=pathfilter):
        print(pathfilter+file.split('/')[-1])
        writevid = WriteVideo(filename=output_path+file.split('/')[-1], frame_size=frame_size)
        for filename in BatchProcess(pathfilter=file[:-4]+"*.mp4", reverse_sort=True):
            readvid=ReadVideo(filename)
            for frame in range(readvid.num_frames):
                writevid.add_frame(readvid.read_next_frame())
            readvid.close()
        writevid.close()

def extract_section_movies_in_folder(pathfilter, output_path, start_frame=0, stop_frame=1, step=1, frame_size=(2048,2048,3)):
    for file in BatchProcess(pathfilter=pathfilter):
        print(pathfilter+file.split('/')[-1])
        writevid = WriteVideo(filename=output_path+file.split('/')[-1], frame_size=frame_size)
        for filename in BatchProcess(pathfilter=file[:-4]+"*.mp4", reverse_sort=True):
            readvid=ReadVideo(filename)
            for frame in range(start_frame,stop_frame,step):
                writevid.add_frame(readvid.read_next_frame())
            readvid.close()
        writevid.close()

if __name__ == '__main__':
    pathfilter='/media/ppzmis/data/ActiveMatter/Microscopy/190820bacteriaand500nmparticles/videos/joined/StreamDIC020.mp4'
    output_path='/media/ppzmis/data/ActiveMatter/Microscopy/190820bacteriaand500nmparticles/videos/joined/test/'
    #concatenate_movies_in_folder(pathfilter, output_path)
    extract_section_movies_in_folder(pathfilter, output_path, start_frame=1, stop_frame=100)
