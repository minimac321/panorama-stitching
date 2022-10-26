import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage


def display_image(img: np.array, title: str = None):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if title is not None:
        plt.title(label=title)


def load_image(img_file_name: str, display_img: bool = False):
    try:
        img = cv2.imread(img_file_name)
        print(f"Found image with shape: {img.shape}")

        if display_img:
            display_image(img)

        return img
    except:
        print(f"Could not read the image: {img_file_name}")
        return None


def save_image(img: np.array, file_name: str):
    if ".jpg" not in file_name and ".png" not in file_name:
        file_name = file_name + ".jpg"

    try:
        cv2.imwrite(file_name, img)
    except Exception as e:
        print(f"Failed to save image to {file_name} with {e}")


def rgb_to_grayscale(img: np.array):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


# Define RGB2gray function
def rgb2gray(img: np.array):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def rgb_to_hsv(img: np.array):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv


def hsv_to_rgb(hsv_img: np.array):
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img


# # Determine gradient function for Fx and Fy using sobel filter(normalized)
# def get_gradient_x(img: np.array):
#     sobel_x_filter = np.array([
#         [-1, 0, 1],
#         [-2, 0, 2],
#         [-1, 0, 1]
#     ])
#     grad_img = ndimage.convolve(img, sobel_x_filter)
#     return grad_img / np.max(grad_img)
#
#
# def get_gradient_y(img: np.array):
#     sobel_y_filter = np.array([
#         [-1, -2, -1],
#         [0, 0, 0],
#         [1, 2, 1],
#     ])
#     grad_img = ndimage.convolve(img, sobel_y_filter)
#     return grad_img / np.max(grad_img)
#
#
# # Define gradient magnitude function
# def gradient_magnitude(fx: np.array, fy: np.array):
#     grad_mag = np.hypot(fx, fy)
#     return grad_mag / np.max(grad_mag)
#
#
# # Define gradient magnitude function
# def gradient_angle(fx: np.array, fy: np.array):
#     grad_angle = np.arctan2(fy, fx)
#     return grad_angle / np.max(grad_angle)
#
#
# # 2.a : Find the closest direction D*
# def closest_dir_function(grad_dir: np.array):
#     closest_dir_arr = np.zeros(grad_dir.shape)
#     for i in range(1, int(grad_dir.shape[0] - 1)):
#         for j in range(1, int(grad_dir.shape[1] - 1)):
#
#             if ((-22.5 < grad_dir[i, j] <= 22.5) or (
#                     -157.5 >= grad_dir[i, j] > 157.5)):
#                 closest_dir_arr[i, j] = 0
#
#             elif ((22.5 < grad_dir[i, j] <= 67.5) or (
#                     -112.5 >= grad_dir[i, j] > -157.5)):
#                 closest_dir_arr[i, j] = 45
#
#             elif ((67.5 < grad_dir[i, j] <= 112.5) or (
#                     -67.5 >= grad_dir[i, j] > -112.5)):
#                 closest_dir_arr[i, j] = 90
#
#             else:
#                 closest_dir_arr[i, j] = 135
#
#     return closest_dir_arr
#
#
# # 2.b : Convert to thinned edge
# def non_maximal_suppressor(grad_mag: np.array, closest_dir: np.array):
#     thinned_output = np.zeros(grad_mag.shape)
#     for i in range(1, int(grad_mag.shape[0] - 1)):
#         for j in range(1, int(grad_mag.shape[1] - 1)):
#
#             if closest_dir[i, j] == 0:
#                 if ((grad_mag[i, j] > grad_mag[i, j + 1]) and (
#                         grad_mag[i, j] > grad_mag[i, j - 1])):
#                     thinned_output[i, j] = grad_mag[i, j]
#                 else:
#                     thinned_output[i, j] = 0
#
#             elif closest_dir[i, j] == 45:
#                 if ((grad_mag[i, j] > grad_mag[i + 1, j + 1]) and (
#                         grad_mag[i, j] > grad_mag[i - 1, j - 1])):
#                     thinned_output[i, j] = grad_mag[i, j]
#                 else:
#                     thinned_output[i, j] = 0
#
#             elif closest_dir[i, j] == 90:
#                 if ((grad_mag[i, j] > grad_mag[i + 1, j]) and (
#                         grad_mag[i, j] > grad_mag[i - 1, j])):
#                     thinned_output[i, j] = grad_mag[i, j]
#                 else:
#                     thinned_output[i, j] = 0
#
#             else:
#                 if ((grad_mag[i, j] > grad_mag[i + 1, j - 1]) and (
#                         grad_mag[i, j] > grad_mag[i - 1, j + 1])):
#                     thinned_output[i, j] = grad_mag[i, j]
#                 else:
#                     thinned_output[i, j] = 0
#
#     return thinned_output / np.max(thinned_output)
#
#
# # Function to include weak pixels that are connected to chain of strong pixels
# def DFS(img: np.array):
#     for i in range(1, int(img.shape[0] - 1)):
#         for j in range(1, int(img.shape[1] - 1)):
#             if (img[i, j] == 1):
#                 t_max = max(img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1], img[i, j - 1],
#                             img[i, j + 1], img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1])
#                 if (t_max == 2):
#                     img[i, j] = 2
#
#
# # Hysteresis Thresholding
# def hysteresis_thresholding(img: np.array, low_ratio: float = 0.10, high_ratio: float = 0.30):
#     diff = np.max(img) - np.min(img)
#     t_low = np.min(img) + low_ratio * diff
#     t_high = np.min(img) + high_ratio * diff
#
#     temp_img = np.copy(img)
#
#     # Assign values to pixels
#     for i in range(1, int(img.shape[0] - 1)):
#         for j in range(1, int(img.shape[1] - 1)):
#             # Strong pixels
#             if (img[i, j] > t_high):
#                 temp_img[i, j] = 2
#             # Weak pixels
#             elif (img[i, j] < t_low):
#                 temp_img[i, j] = 0
#             # Intermediate pixels
#             else:
#                 temp_img[i, j] = 1
#
#     # Include weak pixels that are connected to chain of strong pixels
#     total_strong = np.sum(temp_img == 2)
#     while True:
#         DFS(temp_img)
#         if total_strong == np.sum(temp_img == 2):
#             break
#         total_strong = np.sum(temp_img == 2)
#
#     # Remove weak pixels
#     for i in range(1, int(temp_img.shape[0] - 1)):
#         for j in range(1, int(temp_img.shape[1] - 1)):
#             if temp_img[i, j] == 1:
#                 temp_img[i, j] = 0
#
#     temp_img = temp_img / np.max(temp_img)
#     return temp_img

# def get_pixel(img, row, col, color_band):
#     return img[row][col][color_band]
#
#
# def set_pixel(img, row, col, color_band, value):
#     img[row][col][color_band] = value
#     return img
#
#
# def copy_image(image):
#     img_copy = image.copy()
#     return img_copy




# def shift_image(img, col_band, shift_pch):
#     assert shift_pch <= 1.0
#     assert shift_pch >= 0.0
#     shift_value = shift_pch * 255
#
#     for row in range(img.shape[0]):
#         for col in range(img.shape[1]):
#             pixel_val = get_pixel(img, row, col, col_band)
#             new_pixel_val = pixel_val + shift_value
#
#             set_pixel(img, row, col, col_band, new_pixel_val)
#
#     return img
