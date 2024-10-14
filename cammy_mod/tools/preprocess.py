# %%

import numpy as np
import cv2
import os
import yaml                
from tqdm import tqdm
import numpy as np
from pathlib import Path
import uuid
import matplotlib.pyplot as plt
import json
import argparse
import hashlib

def get_sessions_paths(parent_path, qualifier='lucid', extension='.dat'):
    # look for all .extension files recursively in the parent_path
    sessions = []
    for path in Path(parent_path).rglob(f'**/*{extension}'):
        path = str(path)
        if qualifier.lower() in path.lower():
            sessions.append(path)
    print(f"found {len(sessions)} in {parent_path}")
    return sessions


def click_event(event, x, y, flags, param):
    '''Callback function for mouse events'''
    points, image = param
    # image = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 1000, 0), -1)
        cv2.imshow('image', image)
        if len(points) == 4:
            # If we have 4 points, close the window
            print(points)
            return points


def acquire_points(image):
    '''Acquire points from the user by clicking on the image'''
    points = []

    cv2.imshow('image', image)
    # convert image to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.setMouseCallback('image', click_event, param=(points, image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

def sort_rectangle_points(points):
    '''Sort the points of a rectangle in a clockwise manner'''

    points = np.array(points, dtype=np.float32)

    centroid = np.mean(points, axis=0)

    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]


def readbinaryframe(filepath, start=100, shape=None, preset='lucid'):
    '''Read a frame from a binary file using the start index and shape of the frame
    ARGS:
    filepath: str, path to the binary file
    start: int, index of the frame to read
    shape: tuple, shape of the frame
    preset: str, preset to use for reading the binary file
    RETURNS:
    frame: bytes, the frame read from the file
    '''

    if preset is not None:
        if preset=='lucid':
            # multiply by 2 to accomodate the factt that we save distance data at 16bit
            shape = [640, 480]
            length = np.prod(shape) * 2
            dtype = np.uint16
        elif preset=='flir':
            shape = [720, 540]
            length= np.prod(shape)
            dtype = np.uint8
    else: 
        assert shape != None
        length = np.prod(shape)

    with open(filepath, 'rb') as file:
        file.seek(start*length)  
        frame = file.read(length) 
    return frame, shape, dtype


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


def process_binary(segment, shape, dtype=np.uint16, scaling_factor=0.25):
    frames = np.frombuffer(segment, dtype=dtype)
    frames = frames.reshape(-1, *shape[::-1])
    frames = frames * scaling_factor
    return frames


def remove_artifacts(frame, max_value, floor_value):
    frame = frame.copy()
    frame_mask = frame > max_value
    frame_mask = frame_mask.astype(np.uint16)
    frame[frame_mask==1] = floor_value
    return frame


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


def crop_aligned_rectangle(image, points):
    # Ensure the points are in a NumPy array
    points = np.array(points, dtype=np.float32)

    # Find the bounding box of the points
    rect = cv2.boundingRect(points[:4])
    x, y, w, h = rect

    # Get the transformation matrix for the perspective transform
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points, dst_pts)

    # Apply the perspective transformation to the image
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped



def generate_uuid_from_filename(filepath: str) -> uuid.UUID:
    # Create a SHA-1 hash object
    sha1 = hashlib.sha1()
    
    # Update the hash object with the file path encoded to bytes
    sha1.update(filepath.encode('utf-8'))
    
    # Get the SHA-1 hash of the file path
    hash_bytes = sha1.digest()
    
    # Create a UUID using the first 16 bytes of the SHA-1 hash
    deterministic_uuid = uuid.UUID(bytes=hash_bytes[:16])
    
    return deterministic_uuid



def convert_to_serializable(obj):
    if isinstance(obj, type) and issubclass(obj, np.generic):
        return f"numpy_dtype:{obj.__name__}"
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return obj


def convert_from_serializable(obj):
    if isinstance(obj, str):
        if obj.startswith('numpy_dtype:'):
            dtype_str = obj.split(':')[-1]
            return getattr(np, dtype_str)
        try:
            return uuid.UUID(obj)
        except ValueError:
            pass 
    if isinstance(obj, list):
        return np.array(obj) 
    return obj


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type) and issubclass(obj, np.generic):
            return f"numpy_dtype:{obj.__name__}"
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


def serialize_dict(data):
    return {key: convert_to_serializable(value) for key, value in data.items()}


def deserialize_dict(data):
    return {key: convert_from_serializable(value) for key, value in data.items()}





def get_params(parent_path, colormap='Blues', qualifier='lucid'):
    colormap = 'Blues'
    sessions = get_sessions_paths(parent_path, qualifier=qualifier)

    for session in sessions:
        params = {}
        print(f" Navigating to: {session}")

        # read some frames for visualization and to get the points from the user
        frames, shape, dtype = readbinarysegment(session, start=0, end=10)
        if dtype == np.uint16:
            frame_length = np.prod(shape) * 2
        else:
            frame_length = np.prod(shape)

        # process the binary frame
        frames = process_binary(frames, shape, dtype)
        # show the first frame
        f_ = intensity_to_rgba(frames[0], minval=452, maxval=800, colormap=cv2.COLORMAP_TURBO)
        f = frames[0]
        points = acquire_points(f_)
        # sort the points
        sorted_points = sort_rectangle_points(points)
        cropped = crop_aligned_rectangle(f, sorted_points)
        # define the floor value as the median of the arena
        floor_value = np.median(cropped)
        # add some torelance
        added_margin = np.abs(cropped.mean() - floor_value)
        max_value = floor_value + added_margin
        print('Floor value:', floor_value)
        print('Added margin:', added_margin)
        print('Max tolerated value:', max_value)
        # remove artifacts and replace with floor value
        processed = remove_artifacts(cropped, max_value=max_value, floor_value=floor_value)
        cropped_shape = cropped.shape

        
        # show before and after
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(cropped, vmin=floor_value-100, vmax=floor_value, cmap=colormap)
        ax[1].imshow(processed, vmin=floor_value-100, vmax=floor_value, cmap=colormap)
        plt.show()



        # open a binary file to append the processed frames to it on the fly
        fp = Path(session)
        session_uuid = generate_uuid_from_filename(session)
        output_dir = fp.parent.joinpath(f'session_{session_uuid}.dat').as_posix()
        # print(f"Saving processed frames to {output_dir}")

        # dump the parameters to a yaml file
        params = {
            'input_file': session,
            'output_file': output_dir,
            'input_shape': shape,
            'shape': cropped_shape,
            'dtype': dtype,
            'sorted_points': sorted_points.tolist(),
            'floor_value': floor_value,
            'max_value': max_value,
            'session_uuid': session_uuid
        }

        params = serialize_dict(params)
        with open(fp.parent.joinpath(f'params_{session_uuid}.json').as_posix(), 'w') as config_file:
            json.dump(params, config_file, indent=4, cls=CustomJSONEncoder)


def process_session(params_file, till=None, print_every_n_frames = 1000):

    '''Takes in a params.json file that contains user-input that defines the arena,
    and processes the movie accordingly. The processed frames are saved to a new binary file.'''

    with open(params_file, 'r') as json_file:
        serialized_data = json.load(json_file)
        params = deserialize_dict(serialized_data)

    output_path = params['output_file']
    if till is not None:
        output_path = output_path.replace('.dat', f'_till_{till}.dat')
        params['output_file'] = output_path

    print(' ')
    print(f"starting processing session {params['input_file']}")
    print(f"saving to {params['output_file']}")
    print(' ')
    print("Processing with the following parameters:")
    print('-----------------------------------------')
    for key, value in params.items():
        print(f"{key}: {value}")
        print('')
    print('-----------------------------------------')

    session = params['input_file']
    output_dir = params['output_file']
    shape = params['shape']
    dtype = params['dtype']
    sorted_points = params['sorted_points']
    floor_value = params['floor_value']
    max_value = params['max_value']
    session_uuid = params['session_uuid']
    input_shape = params['input_shape']

    if dtype == np.uint16:
        frame_length = np.prod(input_shape) *2 
    else:
        frame_length = np.prod(input_shape)

    i = 0
    file_length = os.path.getsize(session)
    with open(session, 'rb') as file:
        with open(output_dir, 'wb') as output_file:  
            while True:
                try:
                    if till is not None:
                        if i >= till:
                            break
                    else:
                        if i*frame_length > file_length:
                            break
                    file.seek(i * frame_length)
                    segment = file.read(frame_length)
                    frame = np.frombuffer(segment, dtype=dtype)
                    frame = process_binary(frame, shape=input_shape, dtype=dtype)
                    frame = frame.squeeze()
                    cropped = crop_aligned_rectangle(frame, sorted_points)
                    processed = remove_artifacts(cropped, max_value=max_value, floor_value=floor_value)                              
                    output_file.write(processed.astype(np.uint16).tobytes())
                    # output_file.write(processed.astype(np.uint8))
                    i += 1
                    if i%print_every_n_frames==0:
                        print(f"Processed {i} frames", end='\r')

                except Exception as e:
                    print(e)
                    break


def read_params_and_process(parent_path, till=None):

    '''finds params.json files and uses them to processe the session accordingly.
    parameters mean the points that the user input to indicate the arena, and the shape of the cropped frame. 
    This latter is important because it is necessary information to read the binary file correctly.
    '''
    sessions = get_sessions_paths(parent_path, extension='.json', qualifier='params')
    for session in sessions:
        process_session(params_file=session, till=till)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process raw depth data')
    parser.add_argument('--parent_path', type=str, help='Path to the parent directory containing the raw depth data')
    parser.add_argument('--mode',  type=str, default='process', help='Mode of operation: process or get_params')
    parser.add_argument('--till', type=int, default=None, help='Number of frames to process')
    args = parser.parse_args()

    if args.mode == 'get_params':
        get_params(args.parent_path)
    elif args.mode == 'process':
        read_params_and_process(args.parent_path, till=args.till)



