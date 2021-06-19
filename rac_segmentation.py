import numpy as np
import cv2
from skimage.segmentation import clear_border
from preprocessing import apply_spec_reflection_filter, remove_hair, display_preprocessed_results
from active_contour import region_seg
import matplotlib.pyplot as plt
import os


# Iterative thresholding
def ridler_calvard_threshold(image):
    """
    Finding a threshold value using Ridler and Calvard method

    The reference for this method is:
        "Picture Thresholding Using an Iterative Selection Method"
        by T. Ridler and S. Calvard, in IEEE Transactions on Systems, Man and
        Cybernetics, vol. 8, no. 8, August 1978.
    """
    # Convert image into a 1D array
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
    delta = 0.000000001  # 0
    # Whilst previous threshold and current threshold is not the same, calculate mean for values less than the
    # threshold, mean for values more than the threshold, the current mean is set to the mean of the two means
    while abs(previous_thresh - current_thresh) > delta:
        previous_thresh = current_thresh
        mean1 = np.mean(normalised_img[normalised_img < previous_thresh])
        mean2 = np.mean(normalised_img[normalised_img >= previous_thresh])
        current_thresh = np.mean([mean1, mean2])

    return flatten_img.min() + (flatten_img.max() - flatten_img.min()) * current_thresh


def create_directory(preprocessing, dataset):
    # create directory to store results if it doesn't already exist.
    # separate directory is created for pre-processed images and non-processed images
    if preprocessing:
        result_directory = "RAC steps " + dataset + " preprocessed"
    else:
        result_directory = "RAC steps " + dataset

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    return result_directory


def region_based_active_contour_seg(image, ground_truth, index, dataset, preprocessing):
    if preprocessing:
        # remove hair
        removed_hair = remove_hair(image)

        # apply specular reflection reduction filter
        pre_processed_img = apply_spec_reflection_filter(removed_hair)
        rgb_image = pre_processed_img
        display_preprocessed_results(dataset, image, pre_processed_img, index)
    else:
        rgb_image = np.uint8(image)

    # get rgb channels
    r, g, b = cv2.split(rgb_image)

    # calculate luminance image by combining RGB values according to NTSC 1953 coefficients
    luminance_img = (0.299 * r + 0.587 * g + 0.114 * b)

    # perform thresholding
    threshold = ridler_calvard_threshold(luminance_img)
    im_bw = cv2.threshold(luminance_img, threshold, 255, cv2.THRESH_BINARY)[1]

    inverted = 255 - im_bw
    # create initial mask, this will be used for active contour initialisation
    mask = np.zeros(im_bw.shape)
    contours, h = cv2.findContours(np.uint8(clear_border(inverted)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(inverted, (x, y), (x + w, y + h), (255, 255, 255), 1)
            mask[y:h + y, x:w + x] = 1

    # segment the image
    seg = region_seg(rgb_image, mask)

    # get boundary of ground truth and draw on lesion image
    img = np.uint8(image)
    contours_gt = cv2.findContours(np.uint8(ground_truth), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours_gt) != 0:
        contour_gt = max(contours_gt, key=cv2.contourArea)
        cv2.drawContours(img, [contour_gt], 0, (0, 85, 255), 1)

    contours_automatic = cv2.findContours(np.uint8(seg), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    if len(contours_automatic) != 0:
        for contour in contours_automatic:
            if cv2.contourArea(contour) > 20:
                cv2.drawContours(img, [contour], 0, (56, 255, 76), 1)

    # display and save results
    try:
        result_directory = create_directory(preprocessing, dataset)
        plt.figure(figsize=(13, 5))
        plt.subplot(2, 4, 1)
        plt.imshow(rgb_image)
        plt.title('Lesion Image')
        plt.axis('off')
        plt.subplot(2, 4, 2)
        plt.imshow(luminance_img, cmap='gray')
        plt.title('Luminance Image')
        plt.axis('off')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(im_bw), cmap='gray')
        plt.title('Threshold Image')
        plt.axis('off')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(255 - inverted), cmap='gray')
        plt.title('Initialise contour')
        plt.axis('off')
        plt.subplot(2, 4, 5)
        plt.imshow(rgb_image)
        plt.title('Lesion Image')
        plt.axis('off')
        plt.subplot(2, 4, 6)
        plt.imshow(ground_truth, cmap="gray")
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(2, 4, 7)
        plt.imshow(seg, cmap="gray")
        plt.title('Automatic Segmentation')
        plt.axis('off')
        plt.subplot(2, 4, 8)
        plt.imshow(img, cmap="gray")
        plt.title('GT and automatic border')
        plt.axis('off')
        plt.savefig(result_directory + '/' + str(index), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    except:
        plt.close()

    return seg
