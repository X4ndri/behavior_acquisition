import cv2
import os
import threading
from pathlib import Path


def write_to_txt_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"File written successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def check_avi_frames(avi_paths):
    # Check if paths are valid
    if not avi_paths or len(avi_paths) == 0:
        raise ValueError("The list of AVI paths is empty or None.")
    
    frame_counts = []
    
    for path in avi_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        
        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(frame_count)
        
        cap.release()
    
    # Check if all frame counts are the same
    if len(set(frame_counts)) == 1:
        print(f"All videos have {frame_counts[0]} frames.")
        return True
    else:
        print("Videos do not have the same number of frames.")
        return False


def stack_frames(images_directory, fr=30):
    """ Stacks the .jpg frames acquires from each camera to a single
    .avi file. 
        ARGS: images directory containing cam0 and cam1 images 
    """

    session_name = Path(images_directory).parent
    print(session_name)

    output_path = session_name
    os.system(f'mkdir {output_path}')
    fr = 30

    # Get a list of all image files in the directory
    image_files = sorted([os.path.join(images_directory, file) for file in os.listdir(images_directory) if file.endswith('.jpg')])
    cam0_files = [img for img in image_files if 'cam0' in img]
    cam1_files = [img for img in image_files if 'cam1' in img]

    if len(cam0_files) != len(cam1_files):
        write_to_txt_file(session_name.joinpath("ERROR"), f"cam0 files {len(cam0_files)} not equal to cam1's {len(cam1_files)}")

    print(f'found {len(cam0_files)} cam0 frames')
    print(f'found {len(cam1_files)} cam1 frames')

    # Define the frame size based on the first image in the list
    first_image = cv2.imread(image_files[0])
    frame_size = (first_image.shape[1], first_image.shape[0])

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs too, like 'MJPG' or 'DIVX'
    print(f"saving to: {output_path}")
    left_cam = cv2.VideoWriter(f'{output_path}/left_cam.avi', fourcc, fr, frame_size, isColor=False)
    right_cam = cv2.VideoWriter(f'{output_path}/right_cam.avi', fourcc, fr, frame_size, isColor=False)

    print('Starting... Please do not close this window until instructed!')
    def write_to_avi(writer, images):
        for image in images:
            frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            writer.write(frame)
        writer.release()
        print('thread <--> check')
    cam0_thread = threading.Thread(target=write_to_avi, name='cam0', args=[left_cam, cam0_files])
    cam1_thread = threading.Thread(target=write_to_avi, name='cam1', args=[right_cam, cam1_files])


    cam0_thread.start()
    cam1_thread.start()
    cam0_thread.join()
    cam1_thread.join()

    cv2.destroyAllWindows()
    print(f'Videos created successfully and saved to {output_path}')


if __name__ == '__main__':
    filepath = "/mnt/data_nvme/intercam_nvme/DRD-WT/0513a_a2a_wt/Ldopa/5mg/DRD-WT_0513a_a2a_wt_Ldopa_5mg_v2_300_300_12_20240918110444-728274/side_cameras"
    stack_frames(filepath)