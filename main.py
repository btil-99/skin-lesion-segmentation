from rac_segmentation import region_based_active_contour_seg
import sys
from mean_shift_segmentation import mean_shift_segmentation
import numpy as np
from evaluate import calculate_modified_hausdorff_distance, calc_metrics
from numpy_file_utility import download_save_dataset, load_images
import os

Is_valid = False
Segmentation_type = ""
Dataset_size = 0
Dataset = ""
Preprocessing = False


# Read argument parameters
def read_parameters():
    global Segmentation_type
    global Dataset_size
    global Dataset
    global Preprocessing
    global Is_valid

    # Check whether 4 extra arguments are entered
    if len(sys.argv) == 5:
        # Argument for choosing segmentation method
        if sys.argv[1] != "RAC" and sys.argv[1] != "Mean-shift":
            print("Invalid first parameter. Please select RAC or Mean-shift.")
            Is_valid = False
            return
        else:
            Segmentation_type = sys.argv[1]

        # Argument for choosing dataset type
        if sys.argv[2] != "ISIC" and sys.argv[2] != "PH2":
            print("Invalid second parameter. Please select a dataset (ISIC or PH2)")
            return
        elif sys.argv[2] == "PH2" and not os.path.exists("PH2Dataset/PH2 Dataset images") and not os.path.exists(
                    "PH2Numpy") and not os.path.exists("PH2Dataset.rar"):
            print("Cannot find PH2 data. Please download from the link below and upload 'PH2Dataset.rar' file:")
            print("https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar")
            return
        else:
            Dataset = sys.argv[2]

        # Argument for choosing size of data to run on
        if sys.argv[3].isdigit():
            Dataset_size = int(sys.argv[3])
        else:
            print("Invalid third parameter. Please enter dataset size.")
            return

        # Argument for whether to perform pre-processing
        if sys.argv[4] == "True":
            Preprocessing = True
            Is_valid = True
        elif sys.argv[4] == "False":
            Preprocessing = False
            Is_valid = True
        else:
            print("Invalid option entered for pre-processing. Please select True to pre-process image, and False to "
                  "not perform pre-processing")
            return
    else:
        print("Invalid selections. Please select segmentation type (RAC or Mean-shift), dataset type (ISIC or PH2), "
              "dataset size (int), and whether to perform pre-processing (True or False)")
        Is_valid = False


def perform_segmentation():
    global Dataset_size

    # Download and unzip ISIC data if not already downloaded, and unzip PH2 data if exists
    # Data is then resized and saved as NumPy files, this will improve the speed of getting image data
    # if user wants to re-run the program.
    download_save_dataset()

    # load images from numpy file
    lesion_images, ground_truths = load_images(Dataset + "Numpy")

    # If selected data size is greater than actual length of dataset, use 20% of the data
    if Dataset_size == 0 or Dataset_size > len(lesion_images):
        Dataset_size = round(len(lesion_images) * 0.2)

    lesion_images = lesion_images[:Dataset_size]
    ground_truths = ground_truths[:Dataset_size]

    # numpy array to store segmented results
    segmented_results = np.zeros([Dataset_size, 128, 128])

    # Iterate over images and perform segmentation
    if Segmentation_type == "RAC":
        for i, image in enumerate(lesion_images):
            segmented_result = region_based_active_contour_seg(image, ground_truths[i], i, Dataset, Preprocessing)
            segmented_results[i, :, :] = segmented_result
    else:
        for i, image in enumerate(lesion_images):
            segmented_result = mean_shift_segmentation(image, ground_truths[i], i, Dataset, Preprocessing)
            segmented_results[i, :, :] = segmented_result

    return segmented_results, ground_truths


if __name__ == '__main__':
    read_parameters()
    if Is_valid:
        automatic_borders, ground_truths = perform_segmentation()
        # Evaluate results
        calc_metrics(ground_truths, automatic_borders)
        calculate_modified_hausdorff_distance(ground_truths, automatic_borders)