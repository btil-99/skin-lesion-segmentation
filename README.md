# skin-lesion-segmentation

#### Running mean-shift and region-based implementation ####

Python implementation are stored in SkinLesionSegmentation folder

Requirements:
Python 3.9 is required.

The following libraries will be required to be able to run the python script:
pip install tqdm
pip install requests
pip install sklearn
pip install patool
pip install numpy
pip install imageio
pip install opencv-python
pip install matplotlib
pip install scipy
pip install scikit-image


To run the project, 4 parameters must by given:
- Segmentation type - Whether to run mean-shift clustering implementation, or localised active contour implementation (Mean-shift or RAC)
- Dataset type - Whether to run on ISIC or PH2 dataset (ISIC or PH2)
- Dataset size - How many images to run the implementation on (int)
- Pre-processing - Whether to pre-process images prior segmentation (True or False)

Example of running the project:
python main.py Mean-shift ISIC 5 True
(This will run the mean-shift algorithm on 5 images from the ISIC dataset, the images will be pre-processed prior segmentation) 

If incorrect parameters are entered, a friendly message will be displayed guiding to selecting the 
correct options.

Once the script is running, results will be displayed for 5 seconds and automatically close, closing
the window will automatically start segmenting the next image. Directory files are generated with results,
including a directory for pre-processing ISIC and PH2 images, a directory storing steps/results from Mean-shift
algorithm (one for processed and one with original images). A directory for running active contour segmentation
to store results is also created (a directory for running with pre-processed images, and a directory for running
with original images).

Once it stops running, the border error percentage, TPR, FPR and modified hausdorff distance results
are displayed


#### Running the deep learning implementation ####

In order to run the deep learning implementation, upload BCDU_Net.ipynb file into Google colab 
(https://colab.research.google.com/).

Click on edit at the top of Google colab and select 'Notebook settings'. In the Hardware accelerator, select 
GPU and close it. Then click 'Runtime' at the top of the page and select 'Run all'. This should start 
running the implementation

BCDU_Net.html contains an executed jupyter notebook
