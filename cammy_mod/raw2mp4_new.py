#%%
import cv2
import argparse
import numpy as np
from pathlib import Path


def intensity_to_rgba(frame, minval=452, maxval=3065, colormap=cv2.COLORMAP_TURBO):
    new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
    disp_frame = frame.copy().astype("float")
    disp_frame -= minval
    disp_frame[disp_frame < 0] = 0
    disp_frame /= np.abs(maxval - minval)
    disp_frame[disp_frame >= 1] = 1
    disp_frame *= 255
    bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    new_frame[:, :, :3] = rgb_frame
    new_frame = new_frame.astype(np.uint8)
    return new_frame


def stack2mp4(imgs, output_video_path, dims, fps=30, colormap=cv2.COLORMAP_INFERNO, minval=1200, maxval=2200):
    """
    Convert a sequence of images into an MP4 video.
    
    Args:
    - image_files (list): List of file paths to the input images.
    - output_video_path (str): Path to save the output video.
    - fps (int): Frames per second of the output video.
    """

    print(f'saving to {output_video_path}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, dims, isColor=True)

    for i, im in enumerate(imgs):
        img = intensity_to_rgba(im, minval=minval, maxval=maxval, colormap=colormap)[:,:,:3]
        out.write(img)


    out.release()


def raw2mp4(rawfiledir, dims = [640, 480], fps=30, minval=900, maxval=3065, dtype=np.uint8, colormap=cv2.COLORMAP_INFERNO, preset = None, till=None):
    
    if preset == 'flir':
        minval=50
        maxval = 255
        dims = [720, 540]
        # dims = [360, 540]
        colormap=cv2.COLORMAP_TURBO
        dtype = np.uint8

    if preset == 'lucid':
        minval = 900
        maxval=2300
        dims=[640, 480]
        dtype = np.uint16
        colormap=cv2.COLORMAP_TURBO

    stemname = Path(rawfiledir).stem
    output_dir = Path(rawfiledir).parent.joinpath(f'{stemname}.avi').as_posix()

    # get images
    depth_data = np.fromfile(rawfiledir, dtype=dtype)
    depth_data =  depth_data.reshape([-1, *dims[::-1]])
    a = input(f"found {depth_data.shape[0]} frames. Continue? (y/n): ")
    if a == 'y':
        if till is None:
            stack2mp4(imgs=depth_data, output_video_path=output_dir, fps=fps,dims=dims, minval=minval, maxval=maxval, colormap = colormap)
        else:
            stack2mp4(imgs=depth_data[:till,:,:], output_video_path=output_dir, fps=fps,dims=dims, minval=minval, maxval=maxval, colormap = colormap)   
    else:
        pass

def main():
        print('starting')
        parser = argparse.ArgumentParser(description='Convert a directory of images into an .mp4')
        parser.add_argument('parent_dir', type=str, help='Path to the directory containing the images')
        parser.add_argument('--output_path', type=str, help='Specify output path', default=None)
        parser.add_argument('--preset', type=str, help='specify a preset for minval and maxval pseudocoloring', default=None)
        parser.add_argument('--fps', type=int, help="framerate to save at", default=30)
        parser.add_argument('--min_val', type=int, default=1200, help='Minimum value of the input range')
        parser.add_argument('--max_val', type=int, default=2200, help='Maximum value of the input range')
        parser.add_argument('--till', type=int, default=None, help='consider beginning of recording till this frame')


        args = parser.parse_args()

        raw2mp4(rawfiledir=args.parent_dir, preset=args.preset, fps=args.fps, minval=args.min_val, maxval=args.max_val, till=args.till)


if __name__ == '__main__':
    main()

# %%
# raw2mp4("/home/ahmad/Desktop/behavioral_recordings/intercam/drd_behavior/m41(l3)/none/none/drd_behavior_m41(l3)_none_none_v2_300_300_12_20240424103058-118922/FLIR-1E10011883A9-011883A9.dat", preset='flir', till=100)