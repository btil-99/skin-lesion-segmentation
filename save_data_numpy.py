from download_data import download_and_unzip
import glob
import numpy as np
import imageio
import cv2
import os

height = 128
width = 128
channels = 3

isic_numpy_directory = "ISICNumpy"
ph2_numpy_directory = "PH2Numpy"


def get_isic_dataset():
    dataset_directory = "ISIC18Dataset/"
    lesion_img_directory = "ISIC2018_Task1-2_Training_Input"
    # 2594 images in total, 20% used for testing
    testing_data_size = round(2594 * 0.2)
    lesion_files = glob.glob(dataset_directory + lesion_img_directory + '/*.jpg')[:testing_data_size]
    lesion_images = np.zeros([testing_data_size, height, width, channels])
    ground_truth_images = np.zeros([testing_data_size, height, width])

    for i, image in enumerate(lesion_files):
        img = imageio.imread(image)
        img = np.double(cv2.resize(img, (height, width)))
        lesion_images[i, :, :, :] = img

        # get image ID
        image_id = image[len(image) - 16: len(image) - 4]
        ground_truth_file = dataset_directory + 'ISIC2018_Task1_Training_GroundTruth/' + image_id + '_segmentation.png'
        ground_truth = imageio.imread(ground_truth_file)
        ground_truth = np.double(cv2.resize(ground_truth, (height, width)))
        ground_truth_images[i, :, :] = ground_truth

    return lesion_images, ground_truth_images


def get_ph2_dataset():
    dataset_directory = "PH2Dataset/PH2 Dataset images"
    lesion_images = np.zeros([200, height, width, channels])
    ground_truth_images = np.zeros([200, height, width])

    if os.path.exists("PH2Dataset"):
        data_files = next(os.walk(dataset_directory))[1]
        for i in range(len(data_files)):
            # get lesion image
            img = imageio.imread(dataset_directory + "/" + data_files[i] + "/" + data_files[i] + "_Dermoscopic_Image/" + data_files[i] + ".bmp")
            img = np.double(cv2.resize(img, (height, width)))
            lesion_images[i, :, :, :] = img

            # get ground truth image
            ground_truth = imageio.imread(dataset_directory + "/" + data_files[i] + "/" + data_files[i] + "_lesion/" + data_files[i] + "_lesion.bmp")
            ground_truth = np.double(cv2.resize(ground_truth), height, width)
            ground_truth_images[i, :, :] = ground_truth

    return lesion_images, ground_truth_images


def save_numpy_files(directory, lesion_imgs, ground_truths):
    np.save(directory + '/lesion_images', lesion_imgs)
    np.save(directory + '/ground_truths', ground_truths)


def download_save_dataset():
    download_and_unzip()
    # save ISIC data in numpy files
    if not os.path.exists(isic_numpy_directory):
        os.makedirs(isic_numpy_directory)
        isic_lesion_images, isic_ground_truths = get_isic_dataset()
        save_numpy_files(isic_numpy_directory, isic_lesion_images, isic_ground_truths)

    if not os.path.exists(ph2_numpy_directory) and os.path.exists("PH2Dataset/PH2 Dataset images"):
        os.makedirs(ph2_numpy_directory)
        ph2_lesion_images, ph2_ground_truths = get_ph2_dataset()
        save_numpy_files(ph2_numpy_directory, ph2_lesion_images, ph2_ground_truths)