import os
import cv2
import numpy as np

def compute_homography_from_directory(image_dir, checkerboard_size, square_size):
    """
    Compute the homography matrix for a camera using checkerboard images from a directory.

    Args:
        image_dir (str): Path to the directory containing checkerboard images.
        checkerboard_size (tuple): Number of inner corners per row and column (e.g., (7, 5)).
        square_size (float): Size of a square in the checkerboard (in real-world units, e.g., meters).

    Returns:
        np.ndarray: Homography matrix mapping image points to real-world points.
    """
    # Prepare object points (3D points in the real world)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by the size of a square

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Get all image files in the directory
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for image_path in image_files:
        # Read the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners (optional)
            cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            cv2.imshow("Checkerboard", img)
            cv2.waitKey(500)
        else:
            print(f"Checkerboard not found in {image_path}")

    cv2.destroyAllWindows()

    # Compute the homography matrix using the first valid image
    if len(objpoints) > 0 and len(imgpoints) > 0:
        H, _ = cv2.findHomography(imgpoints[0], objpoints[0][:, :2])
        return H
    else:
        raise ValueError("No valid checkerboard images found.")

if __name__ == "__main__":
    # Example usage
    image_dir = "./checkerboard_images"  # Directory containing checkerboard images
    checkerboard_size = (7, 5)  # Number of inner corners per row and column
    square_size = 0.025  # Size of a square in meters (e.g., 2.5 cm)

    homography = compute_homography_from_directory(image_dir, checkerboard_size, square_size)
    print("Homography matrix:")
    print(homography)

    # Save the homography matrix
    np.save("homography_camera.npy", homography)