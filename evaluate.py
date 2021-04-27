from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    tn, fp, fn, tp = confusion_matrix(groundtruth_list, predicted_list).ravel()

    return tn, fp, fn, tp


def calculate_metrics(mask_data, predictions):
    sum_sensitivity = 0
    sum_specificity = 0
    sum_xor = 0
    sum_hausdorff = 0
    sum_tpr = 0
    sum_fpr = 0

    total_images = predictions.shape[0] # len(predictions)?

