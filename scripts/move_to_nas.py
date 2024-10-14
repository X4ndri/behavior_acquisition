# %%
from tqdm import tqdm
import shutil
from pathlib import Path
from datetime import datetime
from stack import stack_frames, check_avi_frames


# %%
def write_to_txt_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"File written successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def find_last_modified_subdirectory(base_dir):
    base_path = Path(base_dir)
    last_modified_dir = None
    last_modified_time = None

    for d in base_path.rglob('*'):
        if d.is_dir():
            current_mtime = d.stat().st_mtime
            if last_modified_time is None or current_mtime > last_modified_time:
                last_modified_dir = d
                last_modified_time = current_mtime

    return last_modified_dir


# Validate that all files have been copied
def validate_directories(src, dest):
    for src_item in src.rglob('*'):
        if src_item.is_file():
            dest_item = dest / src_item.relative_to(src)
            # Check if the file exists in the destination
            if not dest_item.exists():
                return False
            # Check if the file sizes are the same
            if src_item.stat().st_size != dest_item.stat().st_size:
                return False
            print(f"Successfully copied {src_item} to {dest_item}")
    return True

# %%

def move_directory_preserve_structure(depth_behavior_source,
                                      side_camera_behavior_source, 
                                      calcium_path, 
                                      depth_base_dir,
                                      target_dir):
    depth_source_path = Path(depth_behavior_source)
    side_camera_source_path = Path(side_camera_behavior_source)


    target_base_path = Path(target_dir)
    
    # Validate that all given directories do in fact exist
    if not depth_source_path.exists():
        raise FileNotFoundError(f"DEPTH Source directory {depth_source_path} does not exist.")
    if not side_camera_source_path.exists:
        raise FileNotFoundError(f"SIDE CAMERA Source directory {side_camera_source_path} does not exist")
    if not Path(depth_base_dir).exists():
        raise FileNotFoundError(f"DEPTH Base directory {depth_base_dir} does not exist.")
    if not target_base_path.exists():
        raise FileNotFoundError(f"Target directory {target_base_path} does not exist.")
    

    # move the converted side camera .avi files to the depth location
    side_cam_avi_paths = list(side_camera_source_path.glob('*.avi'))

    if side_cam_avi_paths == []:
        print("NO AVI FILES WERE FOUND FOR SIDE CAMS, TRYING TO CONVERT RAW")
        stack_frames(side_camera_source_path.joinpath('side_cameras').as_posix())
        print("Finished converting to avi")
        print("consolidating side cam and depth items...")
        for item in side_cam_avi_paths:
            dest = depth_behavior_source / item.relative_to(side_camera_behavior_source)
            shutil.move(item, dest)
    else:
        print("consolidating side cam and depth items...")
        VALID = check_avi_frames([str(x) for x in side_cam_avi_paths])
        if not VALID:
            # check if it's inherent to the recording
            if side_camera_source_path.joinpath("ERROR").exists():
                pass
            else:
                print("original avi conversion error, converting again...")
                stack_frames(side_camera_source_path.joinpath('side_cameras').as_posix())
                side_cam_avi_paths = list(side_camera_source_path.glob('*.avi'))

        for item in side_cam_avi_paths:
            dest = depth_behavior_source / item.relative_to(side_camera_behavior_source)
            shutil.move(item, dest)

    # Extract the relative path from the base directory
    relative_path = depth_source_path.relative_to(depth_base_dir)
    print(f"DEPTH Relative Path: {relative_path}")

    # Construct the full target path
    target_path = target_base_path / relative_path
    print(f"Target Path: {target_path}")
    
    # Create necessary directories in the target location
    target_path.mkdir(parents=True, exist_ok=True)

    for i, sp in enumerate([depth_behavior_source, Path(calcium_path)]): 
        print(f"Moving {sp} to {target_path}")
        # Calculate total size for the progress bar
        total_size = sum(f.stat().st_size for f in sp.rglob('*') if f.is_file())
        
        # Initialize the progress bar
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Moving files")
        
        # Move each file in source path to the target path
        for item in sp.rglob('*'):
            dest = target_path / item.relative_to(sp)
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                # Move the file while updating the progress bar
                shutil.copy2(item, dest)
                pbar.update(item.stat().st_size)
        # Close the progress bar
        pbar.close()
    
        if validate_directories(sp, target_path):
            # Clean up the source directory if validation is successful
            # i=1 cooresponds to the behavioral data. We don't want to delete the calcium data 
            # because nvoke does not allow us to delete files. 
            if i==0:
                try:
                    shutil.rmtree(sp)
                except:
                    print("failed to remove original behavior files")
                print("Successfully removed original behavior files")
            print(f"Successfully moved {sp} to {target_path}\n")
            print("REMEMBER TO DELETE THE CALCIUM DATA FROM THE NVOKE.")

        else:
            print("Validation failed: some files may not have been copied correctly.")


def main():

    # find the last modified depth behavior directory (on data_sp)
    last_modified_depth_behavior_subdirectory = find_last_modified_subdirectory(depth_behavior_base_directory)
    # find the last modified side cameras behavior directory (on data_nvme)
    last_modified_side_camera_behavior_subdirectory = find_last_modified_subdirectory(side_cam_base_directory)

    # find the last modified calcium session on the nvoke
    last_modified_calcium_subdirectory = find_last_modified_subdirectory(calcium_base_directory)

    print("THIS SCRIPT WILL MOVE THE LAST MODIFIED BEHAVIOR AND CALCIUM DIRECTORIES TO THE NAS.")
    print('\n\n\n')
    
    print(f"Last modified DEPTH subdirectory: {last_modified_depth_behavior_subdirectory}")
    print('\n')
    print(f"Last modified SIDE CAMERA subdirectory: {last_modified_side_camera_behavior_subdirectory}")
    print('\n')
    print(f"Last modified CALCIUM subdirectory: {last_modified_calcium_subdirectory}")
    print('\n')

    x = input("Read the above directories. Are they correct? (y/n): ")
    if x.lower() != 'y':
        print("Exiting...")
        return

    move_directory_preserve_structure(depth_behavior_source = last_modified_depth_behavior_subdirectory,
                                      side_camera_behavior_source = last_modified_side_camera_behavior_subdirectory, 
                                      calcium_path = last_modified_calcium_subdirectory, 
                                      depth_base_dir = depth_behavior_base_directory,
                                      target_dir = target_dir)

if __name__ == "__main__":
    # Set Paths Here
    depth_behavior_base_directory = "/mnt/data_sp/depth_behavior"
    side_cam_base_directory = "/mnt/data_nvme/side_camera_behavior/"
    calcium_base_directory = "/mnt/nVoke/"
    target_dir = "/home/ahmad/Desktop/datastor/ehlab/Xueliang/calcium_imaging/"

    main()
