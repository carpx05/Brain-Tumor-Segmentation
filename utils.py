from radiomics import featureextractor
import SimpleITK as sitk
import six
import numpy as np
import nibabel as nib


def read_sitk_img(img_path):
    image_data = sitk.ReadImage(img_path)
    return image_data 


def read_nii_img(img_path):
    image_data = np.array(nib.load(img_path).get_fdata())
    return image_data


def centre_of_tumor(input_image,input_mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    result = extractor.execute(input_image,input_mask)
    print(result)
    centre_of_mass = []
    for key, value in six.iteritems(result):
        if key == 'diagnostics_Mask-original_CenterOfMassIndex':
            centre_of_mass.append(value[0])
            centre_of_mass.append(value[1])
            centre_of_mass.append(value[2]) 
    return centre_of_mass


def crop_images(img,centre_of_mass,optimal_roi):
  
    x_roi = round(optimal_roi[0]/2)
    y_roi = round(optimal_roi[1]/2)
    z_roi = round(optimal_roi[2]/2)
    x_centre = round(centre_of_mass[0])
    y_centre = round(centre_of_mass[1])
    z_centre= round(centre_of_mass[2])

    sizeX, sizeY, sizeZ = img.GetSize() # Get image dimensions using GetSize()

    if x_centre - x_roi < 0:
        start_x = 0 
        end_x = optimal_roi[0]
    elif x_centre + x_roi >= sizeX:
        end_x = sizeX - 1
        start_x = end_x - optimal_roi[0]
    else:
        start_x = x_centre - x_roi
        end_x =  x_centre + x_roi

    if y_centre - y_roi < 0:
        start_y = 0 
        end_y = optimal_roi[1]
    elif y_centre + y_roi >= sizeY:
        end_y = sizeY - 1
        start_y = end_y - optimal_roi[1]
    else:
        start_y = y_centre - y_roi
        end_y =  y_centre + y_roi

    if z_centre - z_roi < 0:
        start_z = 0 
        end_z = optimal_roi[2]
    elif z_centre + z_roi >= sizeZ:
        end_z = sizeZ - 1
        start_z = end_z - optimal_roi[2]
    else:
        start_z = z_centre - z_roi
        end_z =  z_centre + z_roi

    cropped_img = img[start_x:end_x,
                        start_y:end_y,
                        start_z:end_z] 
    return cropped_img


def rescale_image(image):
    filter = sitk.RescaleIntensityImageFilter()
    filter.SetOutputMaximum(255)
    filter.SetOutputMinimum(0)
    rescaled_img = filter.Execute(image)
    return rescaled_img


def normalize_image(image):
    image = sitk.GetArrayFromImage(image)
    mask = np.where(image != 0)
    desired_img = image[mask]
    mean = np.mean(desired_img)
    std = np.std(desired_img)
    final_image = (image - mean)/ std
    final_image = sitk.GetImageFromArray(final_image)
    return final_image


def n4_bias_correction(image):
    image = sitk.Cast(image,sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_corrected = corrector.Execute(image)
    return output_corrected