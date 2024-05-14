import cv2
import numpy as np
import os
import string

# Specify the folder with the images to be augmented
folder_path = "D:/SRM/Assingment/8Sem/Review/2nd_git/two/dataSetHarshu/trainingData/0"
save_path = "D:/SRM/Assingment/8Sem/Review/2nd_git/two/dataSetHarshu/trainingData/0"
#for i in string.ascii_uppercase:
#   folder_path = "D:/SRM/Assingment/8Sem/Review/2nd_git/two/dataSetHarshu/trainingData/" + i
#    save_path = "D:/SRM/Assingment/8Sem/Review/2nd_git/two/dataSetHarshu/trainingData/" + i

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        img_gray= img[:,:,1]
        
        #salt and pepper noise
        # Get the image size (number of pixels in the image).
        img_size = img_gray.size

        # Set the percentage of pixels that should contain noise
        noise_percentage = 0.1  # Setting to 10%

        # Determine the size of the noise based on the noise precentage
        noise_size = int(noise_percentage*img_size)

        # Randomly select indices for adding noise.
        random_indices = np.random.choice(img_size, noise_size)

        # Create a copy of the original image that serves as a template for the noised image.
        img_noised = img_gray.copy()

        # Create a noise list with random placements of min and max values of the image pixels.
        noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)

        # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
        img_noised.flat[random_indices] = noise

        new_filename_1 = "noise_" + filename
        new_img_path_1 = os.path.join(save_path, new_filename_1)
        cv2.imwrite(new_img_path_1, img_noised)



# Load an image
'''ima = cv2.imread("B202.jpg")
img_gray= ima[:,:,1]

#salt and pepper noise
# Get the image size (number of pixels in the image).
img_size = img_gray.size

# Set the percentage of pixels that should contain noise
noise_percentage = 0.1  # Setting to 10%

# Determine the size of the noise based on the noise precentage
noise_size = int(noise_percentage*img_size)

# Randomly select indices for adding noise.
random_indices = np.random.choice(img_size, noise_size)

# Create a copy of the original image that serves as a template for the noised image.
img_noised = img_gray.copy()

# Create a noise list with random placements of min and max values of the image pixels.
noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)

# Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
img_noised.flat[random_indices] = noise


#Gausian noise
#noise = np.random.normal(0, 75, img_gray.shape) 
#img_noised = img_gray + noise
#img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

cv2.imshow("Noisy Image", img_noised )
cv2.waitKey(0)
cv2.destroyAllWindows()'''