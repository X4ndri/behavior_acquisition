import os
import cv2
from pathlib import Path
import numpy as np

def read_images(image_folder, camera):
    images = []
    filenames = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.startswith(camera):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
            else:
                print(f"Failed to read image: {img_path}")
    return images, filenames

def detect_synced_charuco_corners(images_cam1, images_cam2, filenames_cam1, filenames_cam2, board, dictionary, view=True):
    assert len(images_cam1) == len(images_cam2), "The number of images from both cameras must be the same."
    synced_corners_cam1 = []
    synced_corners_cam2 = []
    synced_ids_cam1 = []
    synced_ids_cam2 = []
    valid_filenames_cam1 = []
    valid_filenames_cam2 = []

    for img_cam1, img_cam2, filename_cam1, filename_cam2 in zip(images_cam1, images_cam2, filenames_cam1, filenames_cam2):
        gray_cam1 = cv2.cvtColor(img_cam1, cv2.COLOR_BGR2GRAY)
        gray_cam2 = cv2.cvtColor(img_cam2, cv2.COLOR_BGR2GRAY)
        
        corners_cam1, ids_cam1, _ = cv2.aruco.detectMarkers(gray_cam1, dictionary)
        corners_cam2, ids_cam2, _ = cv2.aruco.detectMarkers(gray_cam2, dictionary)

        if ids_cam1 is not None and ids_cam2 is not None:
            _, charuco_corners_cam1, charuco_ids_cam1 = cv2.aruco.interpolateCornersCharuco(corners_cam1, ids_cam1, gray_cam1, board)
            _, charuco_corners_cam2, charuco_ids_cam2 = cv2.aruco.interpolateCornersCharuco(corners_cam2, ids_cam2, gray_cam2, board)

            if (charuco_corners_cam1 is not None and len(charuco_corners_cam1) >= 4 and
                charuco_ids_cam1 is not None and
                charuco_corners_cam2 is not None and len(charuco_corners_cam2) >= 4 and
                charuco_ids_cam2 is not None):
                synced_corners_cam1.append(charuco_corners_cam1)
                synced_corners_cam2.append(charuco_corners_cam2)
                synced_ids_cam1.append(charuco_ids_cam1)
                synced_ids_cam2.append(charuco_ids_cam2)
                valid_filenames_cam1.append(filename_cam1)
                valid_filenames_cam2.append(filename_cam2)

                # Overlay detected corners on the images and show them
                img_cam1_marked = cv2.aruco.drawDetectedCornersCharuco(img_cam1, charuco_corners_cam1, charuco_ids_cam1)
                img_cam2_marked = cv2.aruco.drawDetectedCornersCharuco(img_cam2, charuco_corners_cam2, charuco_ids_cam2)
                if view:
                    cv2.imshow('Camera 1', img_cam1_marked)
                    cv2.imshow('Camera 2', img_cam2_marked)

                    print(f"Frame pair: {filename_cam1}, {filename_cam2} - Press 'q' to continue to the next frame")

                    while True:
                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            break

                    cv2.destroyAllWindows()

            else:
                print(f"Skipping frame pair: {filename_cam1}, {filename_cam2} due to insufficient corners")
        else:
            print(f"Skipping frame pair: {filename_cam1}, {filename_cam2} due to undetected markers")
    
    return synced_corners_cam1, synced_corners_cam2, synced_ids_cam1, synced_ids_cam2, valid_filenames_cam1, valid_filenames_cam2

def calibrate_camera(charuco_corners, charuco_ids, board, image_size):
    if len(charuco_corners) < 1:
        raise ValueError("Not enough corners for calibration")
    
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros((5, 1))
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charuco_corners, charuco_ids, board, image_size, camera_matrix, dist_coeffs)
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def main():
    image_folder = "/home/ahmad/Desktop/behavioral_recordings/intercam/calibration/1/na/na/calibration_1_na_na_v2_300_300_12_20240711203523-693840/selected_calibration_images/"
    camera1_images, camera1_filenames = read_images(image_folder, 'cam1')
    camera2_images, camera2_filenames = read_images(image_folder, 'cam2')

    if len(camera1_images) == 0 or len(camera2_images) == 0:
        print("No images found for one or both cameras.")
        return

    # Define the ChArUco board
    squares_x = 7  # Number of chessboard squares in X direction
    squares_y = 5  # Number of chessboard squares in Y direction
    square_length = 0.01875  # Square length (in meters or any unit you prefer)
    marker_length = 0.01  # Marker side length (in the same unit as square_length)

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

    # Detect corners for both cameras synchronously
    charuco_corners_cam1, charuco_corners_cam2, charuco_ids_cam1, charuco_ids_cam2, _, _ = detect_synced_charuco_corners(
        camera1_images, camera2_images, camera1_filenames, camera2_filenames, board, dictionary)

    if len(charuco_corners_cam1) < 1 or len(charuco_corners_cam2) < 1:
        print("Not enough corners detected for calibration.")
        return

    # Calibrate both cameras
    image_size = camera1_images[0].shape[:2]
    ret_cam1, camera_matrix_cam1, dist_coeffs_cam1, rvecs_cam1, tvecs_cam1 = calibrate_camera(
        charuco_corners_cam1, charuco_ids_cam1, board, image_size)
    ret_cam2, camera_matrix_cam2, dist_coeffs_cam2, rvecs_cam2, tvecs_cam2 = calibrate_camera(
        charuco_corners_cam2, charuco_ids_cam2, board, image_size)

    print("Camera 1 Calibration Results:")
    print("Reprojection Error:", ret_cam1)
    print("Camera Matrix:\n", camera_matrix_cam1)
    print("Distortion Coefficients:\n", dist_coeffs_cam1)

    print("\nCamera 2 Calibration Results:")
    print("Reprojection Error:", ret_cam2)
    print("Camera Matrix:\n", camera_matrix_cam2)
    print("Distortion Coefficients:\n", dist_coeffs_cam2)

    # Save calibration results
    cam1_file = Path(image_folder).joinpath("calibration_cam1.npz")
    cam2_file = Path(image_folder).joinpath("calibration_cam2.npz")
    np.savez(cam1_file, camera_matrix=camera_matrix_cam1, dist_coeffs=dist_coeffs_cam1)
    print(f"saved to: {cam1_file}")
    np.savez(cam2_file, camera_matrix=camera_matrix_cam2, dist_coeffs=dist_coeffs_cam2)
    print(f"saved to: {cam2_file}")

if __name__ == "__main__":
    main()
