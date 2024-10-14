import numpy as np
from pathlib import Path
import cv2


parent_path = "/home/ahmad/Desktop/behavioral_recordings/intercam/calibration/1/na/na/calibration_1_na_na_v2_300_300_12_20240711203523-693840/"


# Load calibration parameters for Camera 1
data_cam1 = np.load(parent_path+'selected_calibration_images/calibration_cam1.npz')
camera_matrix_cam1 = data_cam1['camera_matrix']
dist_coeffs_cam1 = data_cam1['dist_coeffs']

# Load calibration parameters for Camera 2
data_cam2 = np.load(parent_path+'selected_calibration_images/calibration_cam2.npz')
camera_matrix_cam2 = data_cam2['camera_matrix']
dist_coeffs_cam2 = data_cam2['dist_coeffs']



def undistort_image(image_path, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted_image

# read a random image from the specified image dir
image_path = Path(parent_path).joinpath("selected_calibration_images")
pngs = image_path.glob('*.png')
pngs = list(pngs)
random_image = np.random.randint(0, len(pngs))
template = Path(pngs[random_image]).name
template = template.split('_')
template = template[1:]
cam1_img = 'cam1_' + '_'.join(template)
cam2_img = 'cam2_' + '_'.join(template)
cam1_img = image_path.joinpath(cam1_img)
cam2_img = image_path.joinpath(cam2_img)


undistorted_image_cam1 = undistort_image(cam1_img.as_posix(), camera_matrix_cam1, dist_coeffs_cam1)
undistorted_image_cam2 = undistort_image(cam2_img.as_posix(), camera_matrix_cam2, dist_coeffs_cam2)

cv2.imshow('Undistorted Camera 1', undistorted_image_cam1)
cv2.imshow('Undistorted Camera 2', undistorted_image_cam2)
cv2.waitKey(0)
cv2.destroyAllWindows()