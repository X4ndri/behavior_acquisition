from moseq2_extract.io.video import load_movie_data, get_movie_info, write_frames
from os.path import join, exists, dirname, basename, abspath, splitext
from moseq2_extract.util import gen_batch_sequence
from tqdm import tqdm
from pathlib import Path
import numpy as np
import subprocess
import logging
import os
import sys


def get_sessions_paths(parent_path, qualifier='flir', extension='.dat'):
    # look for all .extension files recursively in the parent_path
    sessions = []
    for path in Path(parent_path).rglob(f'**/*{extension}'):
        if qualifier.lower() in path.stem.lower():
            sessions.append(str(path))
    print(f"found {len(sessions)} in {parent_path}")
    return sessions


def readbinarysegment(filepath, start=0, end=100, shape=(640, 480), dtype=np.uint16):
        
    num_frames = end-start
    if dtype == np.uint16:
        frame_length = np.prod(shape) * 2
    else:
        frame_length = np.prod(shape)
    total_length = num_frames * frame_length

    with open(filepath, 'rb') as file:
        file.seek(start*frame_length)  # Move the pointer
        segment = file.read(total_length)  # Read the specfied length
    return segment, shape, dtype


def write_frames(
    filename,
    frames,
    threads=6,
    fps=30,
    pixel_format="gray",
    codec="libx264",
    close_pipe=True,
    pipe=None,
    frame_dtype="uint8",
    slices=24,
    slicecrc=1,
    frame_size=None,
    get_cmd=False,
):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (subProcess.Pipe): pipe to currently open video file.
    frame_dtype (str): indicates the data type to use when writing the videos
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)
    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm(
        range(frames.shape[0]), disable=True, desc=f"Writing frames to {filename}"
    ):
        pipe.stdin.write(frames[i].astype(frame_dtype).tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def process_binary(segment, shape, dtype=np.uint16, scaling_factor=0.25):
    frames = np.frombuffer(segment, dtype=dtype)
    frames = frames.reshape(-1, *shape[::-1])
    frames = frames * scaling_factor
    return frames


def main(parent_path,
         output_file=None,
         frame_size=(720,540),
         movie_dtype='<u1',
         chunk_size=3000,
         mapping="DEPTH",
         threads=4,
         fps=30,
         delete_source=False
         ):
    
    files = get_sessions_paths(parent_path=parent_path)
    i = 0
    for input_file in tqdm(files):
        output_file = None
        try:

            print(f"Starting with; {input_file}")

            log_file = Path(parent_path).joinpath(f"side_cameras_to_avi.log")
            logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
            logging.info(f"Starting with; {input_file}")


            # # make a test dataset to see if conversion is successful
            # segment, shape, dtype = readbinarysegment(input_file, start=0, end=2000, shape=(720,540), dtype=np.uint8)
            # test_dataset_name = input_file.replace('.dat', '_test.dat')
            # with open(test_dataset_name, 'wb') as file:
            #     file.write(segment)
            # print(f"test dataset saved to {test_dataset_name}")
            # input_file = test_dataset_name

            if output_file is None:
                base_filename = splitext(basename(input_file))[0]
                output_file = join(dirname(input_file), f"{base_filename}.avi")
                print(f"output file set to {output_file}")
                logging.info(f"output file set to {output_file}")

            vid_info = get_movie_info(input_file, frame_size=frame_size, mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)
            frame_batches = gen_batch_sequence(vid_info["nframes"], chunk_size, 0)

            video_pipe = None
            for batch in tqdm(frame_batches, desc="Encoding batches", leave=False):
                frames = load_movie_data(input_file, batch, frame_size=tuple(frame_size), mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)

                video_pipe = write_frames(
                    output_file,
                    frames,
                    pipe=video_pipe,
                    close_pipe=False,
                    threads=threads,
                    fps=fps,
                )

            if video_pipe:
                video_pipe.communicate()
            
            # check if the same numbers of frames are in the input and output files
            input_frames = get_movie_info(input_file, frame_size=frame_size, mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)['nframes']
            output_frames = get_movie_info(output_file, frame_size=frame_size, mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)['nframes']
            # check to see if the dims are the same
            input_dims = get_movie_info(input_file, frame_size=frame_size, mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)['dims']
            output_dims = get_movie_info(output_file, frame_size=frame_size, mapping=mapping, bit_depth=8, movie_dtype=movie_dtype)['dims']

            if input_frames == output_frames and input_dims == output_dims:
                print(f"Conversion successful: {input_frames} frames in {input_file} and {output_frames} frames in {output_file}")
                logging.info(f"Conversion successful: {input_frames} frames in {input_file} and {output_frames} frames in {output_file}")
                if delete_source:
                    y = input(f"Delete flag is set, do you still want to proceed? (y/n)")
                    if y == 'y':
                        os.remove(input_file)
                        print(f"Deleted {input_file}")
                        logging.info(f"Deleted {input_file}")
                    else:
                        sys.exit()
            else:
                logging.error(f"Conversion failed: {input_frames} frames in {input_file} and {output_frames} frames in {output_file}")
                raise ValueError(f"Conversion failed: {input_frames} frames in {input_file} and {output_frames} frames in {output_file}")
        except Exception as e:
            pass
        finally:
            i+=1


if __name__ == '__main__':
    main(parent_path='/home/ahmad/Desktop/behavioral_recordings/intercam/calibration/0/na/na/calibration_0_na_na_v2_300_300_12_20240710182715-270048/',
         output_file=None,
         frame_size=(720,540),
         movie_dtype='<u1',
         chunk_size=3000,
         mapping="DEPTH",
         threads=32,
         fps=30,
         delete_source=False
        )