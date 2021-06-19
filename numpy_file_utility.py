from download_data import download_and_unzip
import glob
import numpy as np
import imageio
import cv2
import os
from tqdm import tqdm

# Define dimensions
width = 128
height = 128
channels = 3

# Define Numpy directories
isic_numpy_directory = "ISICNumpy"
ph2_numpy_directory = "PH2Numpy"


def get_isic_dataset():
    dataset_directory = "ISIC18Dataset/"
    lesion_img_directory = "ISIC2018_Task1-2_Training_Input"
    lesion_files = glob.glob(dataset_directory + lesion_img_directory + '/*.jpg')
    lesion_images = np.zeros([len(lesion_files), width, height, channels])
    ground_truth_images = np.zeros([len(lesion_files), width, height])

    for i, image in enumerate(tqdm(lesion_files, desc="Reading ISIC18 data: ")):
        img = imageio.imread(image)
        img = np.double(cv2.resize(img, (width, height)))
        lesion_images[i, :, :, :] = img

        # get image ID
        image_id = image[len(image) - 16: len(image) - 4]
        ground_truth_file = dataset_directory + 'ISIC2018_Task1_Training_GroundTruth/' + image_id + '_segmentation.png'
        ground_truth = imageio.imread(ground_truth_file)
        ground_truth = np.double(cv2.resize(ground_truth, (width, height)))
        ground_truth_images[i, :, :] = ground_truth

    return lesion_images, ground_truth_images


def get_ph2_dataset():
    dataset_directory = "PH2Dataset/PH2 Dataset images"
    lesion_images = np.zeros([200, width, height, channels])
    ground_truth_images = np.zeros([200, width, height])

    if os.path.exists("PH2Dataset"):
        data_files = next(os.walk(dataset_directory))[1]
        for i in range(len(data_files)):
            # get lesion image
            img = imageio.imread(dataset_directory + "/" + data_files[i] + "/" + data_files[i] + "_Dermoscopic_Image/" + data_files[i] + ".bmp")
            img = np.double(cv2.resize(img, (width, height)))
            lesion_images[i, :, :, :] = img

            # get ground truth image
            ground_truth = imageio.imread(dataset_directory + "/" + data_files[i] + "/" + data_files[i] + "_lesion/" + data_files[i] + "_lesion.bmp")
            ground_truth = np.double(cv2.resize(ground_truth, (width, height)))
            ground_truth_images[i, :, :] = ground_truth

    return lesion_images, ground_truth_images


# Save data as numpy files
def save_numpy_files(directory, lesion_imgs, ground_truths):
    np.save(directory + '/lesion_images', lesion_imgs)
    np.save(directory + '/ground_truths', ground_truths)


def download_save_dataset():
    download_and_unzip()
    # save ISIC data in numpy files if NumPy file doesn't already exist
    if not os.path.exists(isic_numpy_directory):
        os.makedirs(isic_numpy_directory)
        isic_lesion_images, isic_ground_truths = get_isic_dataset()
        save_numpy_files(isic_numpy_directory, isic_lesion_images, isic_ground_truths)

    # save PH2 data in numpy files if exists and if NumPy file doesn't already exist
    if not os.path.exists(ph2_numpy_directory) and os.path.exists("PH2Dataset/PH2 Dataset images"):
        os.makedirs(ph2_numpy_directory)
        ph2_lesion_images, ph2_ground_truths = get_ph2_dataset()
        save_numpy_files(ph2_numpy_directory, ph2_lesion_images, ph2_ground_truths)


def load_images(directory):
    lesion_images = np.load(directory + '/lesion_images.npy')
    ground_truths = np.load(directory + '/ground_truths.npy')

    return lesion_images, ground_truths