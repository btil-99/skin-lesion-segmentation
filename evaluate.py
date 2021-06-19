from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.spatial import distance
import cv2


def calc_metrics(ground_truth, predictions):
    y_scores = predictions.reshape(
        predictions.shape[0] * predictions.shape[1] * predictions.shape[2], 1)
    y_true = ground_truth.reshape(ground_truth.shape[0] * ground_truth.shape[1] * ground_truth.shape[2], 1)

    # Threshold
    y_true = np.where(y_true > 0.5, 1, 0)
    y_scores = np.where(y_scores > 0.5, 1, 0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()

    # Calculate border error
    xor = float(fp + fn) / float(tp + fn) * 100

    # Calculate TPR
    tpr = float(tp) / float(tp + fn) * 100

    # Calculate FPR
    fpr = float(fp) / float(fp + tn) * 100

    # Print results
    print("Border Error Metric mean: " + str(xor))
    print("TPR: " + str(tpr))
    print("FPR: " + str(fpr))


def modified_directed_hausdorff(A, B):
    min_euclidean_distance = []
    for a in A:
        euclidean_dist = []
        # Get all euclidean distances from point a to all points in B
        for b in B:
            euclidean_dist.append(distance.euclidean(a, b))

        # Calculate min distance and add it to min_euclidean_distance array
        min_euclidean_distance.append(min(euclidean_dist))

    # Calculate average of min distances
    return np.mean(min_euclidean_distance)


# Since min distance from A to B is not always the same as B to A, the modified directed hausdorff distance is
# calculated from A to B, and from B to A. The mean of the two values is then taken.
def modified_hausdorff_distance(A, B):
    mean_distance = (modified_directed_hausdorff(A, B) + modified_directed_hausdorff(B, A)) / 2

    return mean_distance


# Calculate the overall average hausdorff distance of all images
def calculate_modified_hausdorff_distance(ground_truth, predictions):
    sum_hausdorff = 0

    total_images = predictions.shape[0]

    # Get boundary line for each ground truth image and predicted image. Using those values calculate the modified
    # hausdorff distance
    for i in range(0, total_images):

        contours, hierarchy = cv2.findContours(np.uint8(ground_truth[i]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            ground_truth_border = max(contours, key=cv2.contourArea)

        contours2, hierarchy2 = cv2.findContours(np.uint8(predictions[i]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours2) != 0:
            prediction_border = max(contours2, key=cv2.contourArea)

        hausdorff = modified_hausdorff_distance(ground_truth_border[0, :, :], prediction_border[0, :, :])
        sum_hausdorff += hausdorff

    # Get the average hausdorff distance for all images
    hausdorff_avg = sum_hausdorff / total_images

    print("Modified Hausdorff Distance: " + str(hausdorff_avg))
