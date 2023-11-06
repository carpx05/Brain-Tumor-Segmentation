import numpy as np
import re
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
# import tensorflow as tf
# from scipy import ndimage
# from skimage.exposure import histogram
# from skimage.util import img_as_ubyte
# from skimage import exposure, util
from tensorflow.keras.utils import to_categorical
import os
from utils import read_sitk_img, centre_of_tumor, crop_images, normalize_image, rescale_image, n4_bias_correction
import SimpleITK as sitk


import tarfile
file = tarfile.open('E:/brats dataset 2021/BraTS2021_Training_Data.tar') 

file.extractall('./brain_images')
file.close()
print("\nExtraction done...")

t1_list = sorted(glob.glob('./brain_images/*/*t1.nii.gz'))
t2_list = sorted(glob.glob('./brain_images/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('./brain_images/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('./brain_images/*/*flair.nii.gz'))
mask_list = sorted(glob.glob('./brain_images/*/*seg.nii.gz'))
pattern = re.compile('./brain_images/.*_(\w*)\.nii\.gz')
print("\nLists created...")

optimal_roi = [128,128,128]
image_size = [240,240,155]

for img in range(len(t2_list)):   #Using t2_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
    

    #cropping

    mask = read_sitk_img(mask_list[img])
    image_t1 = read_sitk_img(t1_list[img])
    centre = centre_of_tumor(image_t1,mask)
    cropped_image_t1 = crop_images(image_t1,centre,optimal_roi)
    
    image_t1ce = read_sitk_img(t1ce_list[img])
    cropped_image_t1ce = crop_images(image_t1ce,centre,optimal_roi)
    
    image_t2 = read_sitk_img(t2_list[img])
    cropped_image_t2 = crop_images(image_t2,centre,optimal_roi)
    
    image_flair = read_sitk_img(flair_list[img])
    cropped_image_flair = crop_images(image_flair,centre,optimal_roi)
    
    cropped_mask = crop_images(mask,centre,optimal_roi)
    
    print(type(cropped_image_t1))
    print(type(cropped_image_t1ce))
    print(type(cropped_image_t2))
    print(type(cropped_image_flair))
    print(type(cropped_mask))

    unique_values = np.unique(cropped_mask)
    
    #displaying modularity t1 result, before and after crop
    # fig, ax = plt.subplots(2)
    # ax[0].imshow(sitk.GetArrayViewFromImage(image_t1)[90,:,:], cmap='gray')
    # ax[1].imshow(sitk.GetArrayViewFromImage(cropped_image_t1)[64,:,:], cmap='gray')
    # plt.show()

    print("\nCropping done...")
    


    #rescaling

    rescaled_image_t1 = rescale_image(cropped_image_t1)
    rescaled_image_t1ce = rescale_image(cropped_image_t1ce)
    rescaled_image_t2 = rescale_image(cropped_image_t2)
    rescaled_image_flair = rescale_image(cropped_image_flair)
    rescaled_mask = rescale_image(cropped_mask)
    
    print(type(rescaled_image_t1))
    print(type(rescaled_image_t1ce))
    print(type(rescaled_image_t2))
    print(type(rescaled_image_flair))
    print(type(rescaled_mask))

    print("\nRescaling done...")

    unique_values = np.unique(rescaled_mask)
    


    #normalising
    
    normalized_image_t1 = normalize_image(rescaled_image_t1)
    normalized_image_t1ce = normalize_image(rescaled_image_t1ce)
    normalized_image_t2 = normalize_image(rescaled_image_t2)
    normalized_image_flair = normalize_image(rescaled_image_flair)
    normalized_mask = normalize_image(rescaled_mask)
    print("\nNormalizing done...")
    
    unique_values = np.unique(normalized_mask)


    #image bias correction
    
    bias_corrected_image_t1 = n4_bias_correction(normalized_image_t1)
    bias_corrected_image_t1ce = n4_bias_correction(normalized_image_t1ce)
    bias_corrected_image_t2 = n4_bias_correction(normalized_image_t2)
    bias_corrected_image_flair = n4_bias_correction(normalized_image_flair)
    bias_corrected_mask = n4_bias_correction(normalized_mask)
    print("\nN4 Biasing done...")

    unique_values = np.unique(bias_corrected_mask)
    
    temp_image_t1 = sitk.GetArrayFromImage(bias_corrected_image_t1)
    temp_image_t1ce = sitk.GetArrayFromImage(bias_corrected_image_t1ce)
    temp_image_t2 = sitk.GetArrayFromImage(bias_corrected_image_t2)
    temp_image_flair = sitk.GetArrayFromImage(bias_corrected_image_flair)
    temp_mask = sitk.GetArrayFromImage(bias_corrected_mask)
    
    print(temp_image_t1.shape)
    print(temp_image_t1ce.shape)
    print(temp_image_t2.shape)
    print(temp_image_flair.shape)
    print(temp_mask.shape)



    #combining images and masks  
      
    temp_combined_images = np.stack([temp_image_t1, temp_image_t1ce, temp_image_t2, temp_image_flair], axis=3)
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if not os.path.exists('images'):
        os.makedirs('images')
        
    if not os.path.exists('masks'):
        os.makedirs('masks')
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("\nSaving Image ",img)
        print("temp_mask values:", temp_mask)
        
        if len(temp_mask) > 4:
            temp_mask = temp_mask[:4]

        temp_mask = temp_mask.astype(np.int32)
        print("temp_mask data type:", temp_mask.dtype)
        # Ensure temp_mask values are within the range [0, 3]
        # if np.any(temp_mask < 0) or np.any(temp_mask > 3):
        #     print("Invalid values in temp_mask")
        #     break
        unique_values = np.unique(temp_mask)
        valid_range = np.arange(4)  # 0, 1, 2, 3
        if not np.all(np.isin(unique_values, valid_range)):
            print("Invalid values in temp_mask:", unique_values)
        else:
            temp_mask= to_categorical(temp_mask, num_classes=4)
            np.save('C:/Users/ayush/Documents/Brain-Tumor-Segmentation/images/image_'+str(img)+'.npy', temp_combined_images)
            np.save('C:/Users/ayush/Documents/Brain-Tumor-Segmentation/masks/mask_'+str(img)+'.npy', temp_mask)
            print("Image saved succesfully...")
        
    else:
        print("\nNo useful volume found. Discarding image ", img)
        
    print("\nGoing to next image...\n")
