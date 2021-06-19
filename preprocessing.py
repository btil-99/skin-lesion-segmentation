import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os


def remove_hair(image):
    image = np.uint8(image)
    # convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # create kernel for morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # perform blackHat filtering to find hair strands
    blackhat_filtering = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

    # perform thresholding to intensify hair strands
    ret, thresh = cv2.threshold(blackhat_filtering, 10, 255, cv2.THRESH_BINARY)

    # perform inpainting to remove hair
    processed_img = cv2.inpaint(image, np.uint8(thresh), 1, cv2.INPAINT_TELEA)

    return processed_img


def apply_spec_reflection_filter(image):
    # store hsv information
    image = np.float32(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    # Convert rgb image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply homomorphic filer
    homo_filter_image = homomorphic_filter(gray_image)

    # calculate gamma
    mid = 0.5
    mean = np.mean(gray_image)
    gamma = math.log(mid * 255) / math.log(mean)

    # combine h and s colour channels, and values from applied filter to get RGB image with applied filter
    new_hsv_image = cv2.merge([h, s, homo_filter_image.astype(np.float32)])
    new_rgb = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB) ** gamma
    # normalise rgb values
    new_rgb = (new_rgb - new_rgb.min()) / (new_rgb.max() - new_rgb.min())

    return new_rgb


def homomorphic_filter(image):
    # take the image to log domain and then to frequency domain
    image_log = np.log1p(1 + np.array(image, dtype="uint8"))
    image_fft = np.fft.fft2(image_log)

    # filter
    H = butterworth_filter(image_fft.shape)

    # Apply filter on frequency domain then take the image back to spatial domain
    image_fft_filt = H * image_fft

    # Take the inverse Fourrier transform
    image_filt = abs(np.fft.ifft2(image_fft_filt))

    # Take the exponent transform
    im_n = np.exp(image_filt) - 1
    return im_n


# Butterworth filter calculation
def butterworth_filter(image):
    hl = 0.5
    hh = 1.4
    center_x = image[0] / 2
    center_y = image[1] / 2
    d = 5

    u, v = np.meshgrid(range(image[0]), range(image[1]), sparse=False, indexing='ij')
    distance = (((u - center_x) ** 2 + (v - center_y) ** 2)**0.5).astype(float)

    with np.errstate(divide='ignore'):
        H = (hh - hl) * (1 - np.exp(-butterworth(u, v) * (d / distance))) + hl

    return H


def butterworth(u, v):
    alpha = 1.6
    return (1 + (((u / 2) ** 2 + (v / 2) ** 2) ** 0.5) ** alpha) ** -1


# Display the original image and the image after hair removal and illumination correction
def display_preprocessed_results(dataset_preprocessed, image, preprocessed, index):
    directory = dataset_preprocessed + " preprocessed"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Display results
    try:
        plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.uint8(image))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(preprocessed)
        plt.title('Preprocessed Image')
        plt.axis('off')

        plt.savefig(directory + '/' + str(index), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    except:
        plt.close()
