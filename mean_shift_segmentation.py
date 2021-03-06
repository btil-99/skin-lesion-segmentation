import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
import cv2
import scipy.cluster
from skimage import measure
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
from preprocessing import apply_spec_reflection_filter, remove_hair, display_preprocessed_results


def mean_shift_colour_clustering(image):
    flat_img = np.reshape(image, [-1, 3])

    # get dimension of the neighbourhood
    bandwidth = estimate_bandwidth(flat_img, quantile=0.04)

    # perform clustering on 1D image and return cluster labels
    ms_clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(flat_img)

    # get coordinates of cluster centers
    cluster_centers = ms_clustering.cluster_centers_

    # get clustered image
    clustered_img = cluster_centers[np.reshape(ms_clustering.labels_, image.shape[:2])]

    return clustered_img


# Remove healthy surrounding skin, leaving only the skin lesion
def suppress_skin(image, clustered_image):
    image_shape = (image.shape[0], image.shape[1])
    img = image.copy()
    img[np.all(img < (15, 15, 15), axis=-1)] = (255, 255, 255)
    # rotate blank image to create corner mask
    white_image = np.array(Image.new('RGB', image_shape, color=(255, 255, 255)).rotate(45))

    # adaptive thresholding
    bw_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    img_blur = cv2.blur(bw_img, (9, 9))
    adapt_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1)

    # get corner mask
    inv = np.invert(adapt_thresh)
    corner = white_image[:, :, 0] + inv
    corner_mask = (corner == 0)

    # get unique colours in the corners (colours will represent healthy skin)
    colours = np.unique(clustered_image[corner_mask], axis=0)  # colours in angles

    # count occurrence of each colour from the angles (healthy skin colours) in the corners
    vecs = scipy.cluster.vq.vq(clustered_image[corner_mask], colours)[0]
    colour_counts = np.histogram(vecs, len(colours))[0]

    # remove colours which have been classified as healthy skin colours
    suppressed_skin_img = clustered_image.copy()
    for i in range(len(colour_counts)):
        if colour_counts[i] > 30:
            suppressed_skin_img[np.all(suppressed_skin_img == colours[i], axis=-1)] = (255, 255, 255)

    binary_img = suppressed_skin_img.copy()
    binary_img[binary_img < 255] = 0
    # set colours outside of corner mask/region of interest to white
    binary_img[corner_mask] = (255, 255, 255)

    # apply labelling algorithm and get healthy skin regions to discard
    labels_mask = measure.label(np.invert(np.uint8(binary_img)))
    regions = measure.regionprops(labels_mask)

    # get center of lesion image, this will be used to calculate how far regions are from center
    image_center = np.asarray(binary_img.shape) / 2
    image_center = tuple(image_center.astype('int32'))

    # get regions away from the center of lesion image, these regions will be discarded
    lesions = []
    if len(regions) > 1:
        for region in regions:
            if region.area > 200:
                distance_to_center = (distance.euclidean(image_center, region.centroid))
                lesions.append({'region': region, 'distance_to_center': distance_to_center})
            else:
                labels_mask[region.coords[:, 0], region.coords[:, 1]] = 0
        sorted_lesions = sorted(lesions, key=lambda i: i['distance_to_center'])
        skin_regions = sorted_lesions[1:]

        # discard healthy skin regions
        for region in skin_regions:
            labels_mask[region['region'].coords[:, 0], region['region'].coords[:, 1]] = 0

    suppressed_skin_img[labels_mask == 0] = 255

    return suppressed_skin_img


# Chain code algorithm to extract boundary from image
def extract_boundary(image):
    image_shape = (image.shape[0], image.shape[1])

    gray_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
    contours = cv2.findContours(np.uint8(~gray_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    boundary_img = np.array(Image.new('RGB', image_shape, color=(255, 255, 255)))
    cv2.drawContours(boundary_img, contours, 0, (0, 0, 0), 1)

    return boundary_img, contours


def mean_shift_segmentation(image, ground_truth, index, dataset, preprocessing):
    if preprocessing:
        # remove hair
        removed_hair = remove_hair(image)

        # apply specular reflection reduction filter
        pre_processed_img = apply_spec_reflection_filter(removed_hair)
        rgb_image = np.uint8(pre_processed_img * 255)
        display_preprocessed_results(dataset, image, pre_processed_img, index)
    else:
        rgb_image = np.uint8(image)

    clustered_img = mean_shift_colour_clustering(rgb_image)
    segmented_img = suppress_skin(rgb_image, clustered_img)
    # get boundary border
    boundary_img, contours = extract_boundary(segmented_img)

    # draw boundary on lesion image
    img = np.uint8(image)
    cv2.drawContours(img, contours, 0, (56, 255, 76), 1)

    # get boundary of ground truth and draw on lesion image
    contours_gt = cv2.findContours(np.uint8(ground_truth), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours_gt) != 0:
        contour_gt = max(contours_gt, key=cv2.contourArea)
    cv2.drawContours(img, [contour_gt], 0, (0, 85, 255), 1)

    # get filled segmentation
    filled_boundary = np.zeros((128, 128))
    cv2.fillPoly(filled_boundary, pts=contours, color=(255, 255, 255))

    # create directory to store results if it doesn't already exist
    if preprocessing:
        result_directory = "Mean-shift steps " + dataset + " preprocessed"
    else:
        result_directory = "Mean-shift steps " + dataset
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Automatically close plot display after 3 seconds, if user closes the plot manually, prevents exception and allow
    # user to close
    try:
        plt.figure(figsize=(13, 5))
        plt.subplot(2, 4, 1)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(2, 4, 2)
        plt.imshow(np.uint8(clustered_img))
        plt.title('Clustered Image')
        plt.axis('off')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(segmented_img))
        plt.title('Skin suppression')
        plt.axis('off')
        plt.subplot(2, 4, 4)
        plt.imshow(boundary_img, cmap='gray')
        plt.title('Boundary extraction')
        plt.axis('off')
        plt.subplot(2, 4, 5)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(ground_truth), cmap="gray")
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(filled_boundary), cmap="gray")
        plt.title('Automatic border')
        plt.axis('off')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(img))
        plt.title('GT and automatic border')
        plt.axis('off')
        plt.savefig(result_directory + '/' + str(index), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    except:
        plt.close()

    return filled_boundary
