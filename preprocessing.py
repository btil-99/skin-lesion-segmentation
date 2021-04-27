import numpy as np
import cv2


def remove_hair(image):
    # convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # create kernel for morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # perform blackHat filtering to find hair strands
    blackhat_filtering = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

    # perform thresholding to intensify hair strands
    ret, thresh = cv2.threshold(blackhat_filtering, 5, 255, cv2.THRESH_BINARY)

    # perform inpainting to remove hair
    processed_img = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

    return processed_img


def apply_spec_reflection_filter(image):
    # take the image to log domain and then to frequency domain
    image_log = np.log1p(1 + np.array(image, dtype="uint8"))
    image_fft = np.fft.fft2(image_log)

    # filter
    H = butterworth_filter_test(image_fft.shape)

    # Apply filter on frequency domain then take the image back to spatial domain
    image_fft_filt = H * image_fft

    image_filt = abs(np.fft.ifft2(image_fft_filt))

    im_n = np.exp(image_filt) - 1
    return im_n


def butterworth_filter_test(image):
    hl = 0.5
    hh = 1.4
    center_x = image[0] / 2
    center_y = image[1] / 2
    d = 5

    u, v = np.meshgrid(range(image[0]), range(image[1]), sparse=False, indexing='ij')
    distance = ((u - center_x) ** 2 + (v - center_y) ** 2).astype(float)

    H = (hh - hl) * (1 - np.exp(-butterworth(u, v) * (d / distance))) + hl;
    return H


def butterworth(u, v):
    alpha = 1.6
    return (1 + (((u / 2) ** 2 + (v / 2) ** 2) ** 0.5) ** alpha) ** -1
