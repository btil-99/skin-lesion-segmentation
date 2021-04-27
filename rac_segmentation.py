import numpy as np
import math
import cv2
from skimage.segmentation import clear_border
from preprocessing import apply_spec_reflection_filter, remove_hair
from active_contour import region_seg
import matplotlib.pyplot as plt


def save_hsv_info(image):
    test = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    test = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
    return test


def ridler_calvard_threshold(image):
    """
    Finding a threshold value using Ridler and Calvard method

    The reference for this method is:
        "Picture Thresholding Using an Iterative Selection Method"
        by T. Ridler and S. Calvard, in IEEE Transactions on Systems, Man and
        Cybernetics, vol. 8, no. 8, August 1978.
    """
    flatten_img = np.array(image.flat)

    # The dynamic range of the image is limited. An image with
    # almost all values near zero can give a bad thresholding result.
    min_val = np.max(flatten_img) / 256
    flatten_img[flatten_img < min_val] = min_val

    # normalise image to range between 0-1
    normalised_img = (flatten_img - flatten_img.min()) / (flatten_img.max() - flatten_img.min())
    previous_thresh = 0

    # Mean image intensity is used to get an initial threshold value to start iteration
    current_thresh = np.mean(normalised_img)
    #delta = 0.00001 #0 - test with 0
    delta = 0.000000001
    while abs(previous_thresh - current_thresh) > delta:
        previous_thresh = current_thresh
        mean1 = np.mean(normalised_img[normalised_img < previous_thresh])
        mean2 = np.mean(normalised_img[normalised_img >= previous_thresh])
        current_thresh = np.mean([mean1, mean2])

    return flatten_img.min() + (flatten_img.max() - flatten_img.min()) * current_thresh


def region_based_active_contour_seg(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (450, 360))

    # store hsv information
    hsv_image = save_hsv_info(img.astype(np.float32))
    h, s, v = cv2.split(hsv_image)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    homo_filter_image = apply_spec_reflection_filter(gray_image)

    # calculate gamma
    mid = 0.5
    mean = np.mean(gray_image)
    gamma = math.log(mid * 255) / math.log(mean)

    # combine h and s colour channels, and values from applied filter
    new_hsv_image = cv2.merge([h, s, homo_filter_image.astype(np.float32)])
    new_rgb = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB) ** gamma
    new_rgb = (new_rgb - new_rgb.min()) / (new_rgb.max() - new_rgb.min())

    new_rgb = remove_hair(np.uint8(255 * new_rgb))

    # get rgb channels
    r, g, b = cv2.split(new_rgb)

    # calculate luminance image by combining RGB values according to NTSC 1953 coefficients
    luminance_img = (0.299 * r + 0.587 * g + 0.114 * b)

    threshold = ridler_calvard_threshold(luminance_img)
    im_bw = cv2.threshold(luminance_img, threshold, 255, cv2.THRESH_BINARY)[1]
    inverted = 255 - im_bw

    contours, h = cv2.findContours(np.uint8(clear_border(inverted)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)

    mask = np.zeros(im_bw.shape)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(inverted, (x, y), (x + w, y + h), (255, 255, 255), 2)
            mask[y:h + y, x:w + x] = 1
    """x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(inverted, (x, y), (x + w, y + h), (255, 255, 255), 2)
    mask = np.zeros(im_bw.shape)
    mask[y:h + y, x:w + x] = 1"""

    plt.figure(figsize=(16, 16))
    plt.subplot(1, 4, 1)
    plt.imshow(new_rgb)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(luminance_img), cmap='gray')
    plt.title('Luminance Image')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(im_bw), cmap='gray')
    plt.title('Threshold Image')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(255-inverted, cmap='gray')
    plt.title('Initialise contour')
    plt.axis('off')


    seg = region_seg(new_rgb, mask)

    """plt.figure(1)
    plt.imshow(luminance_img, cmap="gray")

    plt.figure(2)
    plt.imshow(new_rgb)

    plt.figure(3)
    plt.imshow(inverted, cmap="gray")"""

    """plt.figure(4)
    plt.imshow(gray_image2, cmap="gray")"""

    """plt.figure(4)
    plt.imshow(seg)"""

    plt.show()
