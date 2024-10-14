from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def get_sessions_paths(parent_path, qualifier='flir', extension='.dat'):
    # look for all .extension files recursively in the parent_path
    sessions = []
    for path in Path(parent_path).rglob(f'**/*{extension}'):
        if qualifier.lower() in path.stem.lower():
            sessions.append(str(path))
    print(f"found {len(sessions)} in {parent_path}")
    return sessions


# def readbinarysegment(filepath, start=0, end=100, shape=(640, 480), dtype=np.uint16):
        
#     num_frames = end-start
#     if dtype == np.uint16:
#         frame_length = np.prod(shape) * 2
#     else:
#         frame_length = np.prod(shape)
#     total_length = num_frames * frame_length

#     with open(filepath, 'rb') as file:
#         file.seek(start*frame_length)  # Move the pointer
#         segment = file.read(total_length)  # Read the specfied length
#     return segment, shape, dtype



def readbinarysegment(filepath, start=0, end=None, shape=(640, 480), dtype=np.uint16):
        
    if dtype == np.uint16:
        frame_length = np.prod(shape) * 2
    else:
        frame_length = np.prod(shape)
    
    with open(filepath, 'rb') as file:
        if end is None:
            file.seek(0, 2)  # Move the pointer to the end of the file
            file_size = file.tell()  # Get the size of the file
            num_frames = (file_size - start * frame_length) // frame_length
        else:
            num_frames = end - start
        
        total_length = num_frames * frame_length
        file.seek(start * frame_length)  # Move the pointer to the start
        segment = file.read(total_length)  # Read the specified length
        
    return segment, shape, dtype

def process_binary(segment, shape, dtype=np.uint16, scaling_factor=0.25):
    frames = np.frombuffer(segment, dtype=dtype)
    frames = frames.reshape(-1, *shape[::-1])
    frames = frames * scaling_factor
    return frames

class ImageSelector:
    def __init__(self, images_cam1, images_cam2, output_dir):
        self.images_cam1 = images_cam1
        self.images_cam2 = images_cam2
        self.selected_indices = []
        self.output_dir = output_dir

        # Create a figure and two axes for side-by-side display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.image_display_cam1 = self.ax1.imshow(self.images_cam1[0], cmap='gray')
        self.image_display_cam2 = self.ax2.imshow(self.images_cam2[0], cmap='gray')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initial display
        self.index = 0
        self.update_image(self.index)
        
        # Show the plot
        plt.show()

    def update_image(self, index):
        self.image_display_cam1.set_data(self.images_cam1[index])
        self.ax1.set_title(f'Camera 1 - Image Index: {index}')
        self.image_display_cam2.set_data(self.images_cam2[index])
        self.ax2.set_title(f'Camera 2 - Image Index: {index}')
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'enter':
            self.selected_indices.append(self.index)
            print(f'Selected Index: {self.index}')
        elif event.key == 'left':
            self.index = max(0, self.index - 1)
            self.update_image(self.index)
        elif event.key == 'right':
            self.index = min(len(self.images_cam1) - 1, self.index + 1)
            self.update_image(self.index)
    
    def get_selected_indices(self):
        return self.selected_indices

    def save_selected_images(self):
        for idx in self.selected_indices:
            image_cam1 = Image.fromarray(self.images_cam1[idx])
            image_cam2 = Image.fromarray(self.images_cam2[idx])
            image_cam1.save(os.path.join(self.output_dir, f'cam1_image_{idx}.png'))
            image_cam2.save(os.path.join(self.output_dir, f'cam2_image_{idx}.png'))
        print(f"Saved {len(self.selected_indices)} images from each camera to {self.output_dir}")

def main():
    parent_path = "/home/ahmad/Desktop/behavioral_recordings/intercam/calibration/1/na/na/calibration_1_na_na_v2_300_300_12_20240711203523-693840/"
    filepaths = get_sessions_paths(parent_path=parent_path, qualifier='flir')
    
    segment1, shape1, dtype1 = readbinarysegment(filepaths[0], start=0, end=None, shape=(720, 540), dtype=np.uint8)
    frames_cam1 = process_binary(segment=segment1, shape=shape1, dtype=dtype1, scaling_factor=1)
    
    segment2, shape2, dtype2 = readbinarysegment(filepaths[1], start=0, end=None, shape=(720, 540), dtype=np.uint8)
    frames_cam2 = process_binary(segment=segment2, shape=shape2, dtype=dtype2, scaling_factor=1)
    
    ims = ImageSelector(frames_cam1, frames_cam2, output_dir=Path(parent_path).joinpath("selected_calibration_images"))
    print(ims.get_selected_indices())
    ims.save_selected_images()

if __name__ == "__main__":
    main()