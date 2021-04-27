import glob
from rac_segmentation import region_based_active_contour_seg
import sys
from mean_shift_segmentation import mean_shift_segmentation
from download_data import download_and_unzip
import numpy as np
import matplotlib.pyplot as plt


def load_images(directory):
    lesion_images = np.load(directory + '/lesion_images.npy')
    ground_truths = np.load(directory + '/ground_truths.npy')

    return lesion_images, ground_truths


def display_meanshift_results(img, clustered_img, segmented_img, boundary_img):
    f = plt.figure()
    f.add_subplot(1, 4, 1)
    plt.imshow(img)
    f.add_subplot(1, 4, 2)
    plt.imshow(np.uint8(clustered_img))
    f.add_subplot(1, 4, 3)
    plt.imshow(np.uint8(segmented_img))
    f.add_subplot(1, 4, 4)
    plt.imshow(boundary_img, cmap='gray')
    # plt.savefig('1.png', dpi=300, bbox_inches='tight')
    plt.show()


def read_parameters():
    if len(sys.argv) > 1:
        if sys.argv[1] == "RAC":
            print(sys.argv[1])
            download_and_unzip()
            testing_data_size = round(2594 * 0.2)
            #isic_lesion_files = glob.glob("ISIC18Dataset/ISIC2018_Task1-2_Training_input/*.jpg")[:testing_data_size]
            isic_lesion_files = glob.glob("ISIC_0000031.jpg")
            for image in isic_lesion_files:
                region_based_active_contour_seg(image)
        elif sys.argv[1] == "Mean-shift":
            print(sys.argv[1])
            download_and_unzip()

            testing_data_size = round(2594 * 0.2)
            isic_lesion_files = glob.glob("ISIC18Dataset/ISIC2018_Task1-2_Training_input/*.jpg")[:testing_data_size]
            isic_ground_truths_files = glob.glob("ISIC18Dataset/ISIC2018_Task1_Training_GroundTruth/*.png")[
                                       :testing_data_size]

            segmented_results = []
            for i, image in enumerate(isic_lesion_files):
                segmented_result = mean_shift_segmentation(image)
                segmented_results.append(segmented_result)
                # img = cv2.imread(isic_ground_truths_files[i])

            np.save('meanshift_isic_segmented_results', segmented_results)
            # results = np.load('segmented_results.npy')
            """for i in range(len(results)):
                plt.imshow(results[i], cmap="gray")
                plt.show()"""
        else:
            print("No valid option selected. Please select RAC or Mean-shift")
    else:
        print("No valid option selected. Please select RAC or Mean-shift")


if __name__ == '__main__':
    read_parameters()
