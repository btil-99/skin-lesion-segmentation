import cv2
import matplotlib.pyplot as plt

src = cv2.imread("ISIC18Dataset/ISIC2018_Task1-2_Training_input/ISIC_0000451.jpg")

print( src.shape )
image = cv2.resize(src, (450, 360))
print(image.shape)
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# Convert the original image to grayscale
grayScale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Kernel for the morphological filtering
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17,17))

# Perform the blackHat filtering on the grayscale image to find the
# hair countours
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

# intensify the hair countours in preparation for the inpainting
# algorithm
ret,thresh2 = cv2.threshold(blackhat,5,255,cv2.THRESH_BINARY)

# inpaint the original image depending on the mask
dst = cv2.inpaint(rgb,thresh2,1,cv2.INPAINT_TELEA)

"""plt.figure(1)
plt.imshow(rgb)

plt.figure(2)
plt.imshow(grayScale, cmap="gray")

plt.figure(3)
plt.imshow(blackhat, cmap="gray")

plt.figure(4)
plt.imshow(dst)

plt.show()"""


plt.figure(figsize=(16, 16))
plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(dst)
plt.title('Removed Hair')
plt.axis('off')

plt.show()