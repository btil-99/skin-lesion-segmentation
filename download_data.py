from tqdm import tqdm
import requests
import os
import patoolib
import zipfile


def download_data(url, directory_name):
    response = requests.get(url, stream=True)  # HTTP GET Request for url
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(directory_name, 'wb') as file:
        for data in response.iter_content(block_size):
            # File too large for singular download so stream the download in chunks
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    # Finally, close the progress bar
    # Error check
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def unzip_file(folder):
    if os.path.exists(folder):
        if folder.endswith('.rar'):
            patoolib.extract_archive(folder)
        elif folder.endswith('.zip'):
            with zipfile.ZipFile(folder, 'r') as zip_ref:
                zip_ref.extractall('ISIC18Dataset')
        os.remove(folder)


def download_and_unzip():
    zip_location_lesion_imgs = 'isic-image_data.zip'
    zip_location_mask_imgs = 'isic-mask_data.zip'

    isic_lesion_images_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip'
    isic_gt_masks_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip'

    if not os.path.exists(zip_location_lesion_imgs) and not os.path.exists('ISICNumpy') and not os.path.exists('ISIC18Dataset'):
        download_data(isic_lesion_images_url, zip_location_lesion_imgs)
        download_data(isic_gt_masks_url, zip_location_mask_imgs)

    unzip_file(zip_location_lesion_imgs)
    unzip_file(zip_location_mask_imgs)
    unzip_file('PH2Dataset.rar')
