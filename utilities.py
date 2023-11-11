import glob
import re
import SimpleITK as sitk
import numpy as np
import nibabel as nib

def create_path(address, train=False, aug=False, val=False):
    if train:
        t1_pattern = address + '/*GG/*/*t1.nii.gz'
        t2_patern = address + '/*GG/*/*t2.nii.gz'
        flair_pattern = address + '/*GG/*/*flair.nii.gz'
        t1ce_pattern = address + '/*GG/*/*t1ce.nii.gz'
        seg_pattern = address + '/*GG/*/*seg.nii.gz'  # Ground truth
    elif aug:
        t1_pattern = address + '/*GG/*/*/*t1.nii.gz'
        t2_patern = address + '/*GG/*/*/*t2.nii.gz'
        flair_pattern = address + '/*GG/*/*/*flair.nii.gz'
        t1ce_pattern = address + '/*GG/*/*/*t1ce.nii.gz'
        seg_pattern = address + '/*GG/*/*/*seg.nii.gz'
    elif val:
        t1_pattern = address + '/*/*t1.nii.gz'
        t2_patern = address + '/*/*t2.nii.gz'
        flair_pattern = address + '/*/*flair.nii.gz'
        t1ce_pattern = address + '/*/*t1ce.nii.gz'
    t1 = glob.glob(t1_pattern)
    t2 = glob.glob(t2_patern)
    flair = glob.glob(flair_pattern)
    t1ce = glob.glob(t1ce_pattern)
    if not val:
        seg = glob.glob(seg_pattern)
    pattern = re.compile('.*_(\w*)\.nii\.gz')
    if not val:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in
                      list(zip(t1, t2, t1ce, flair, seg))]
    else:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in list(zip(t1, t2, t1ce, flair))]
    return data_paths


def read_sitk_img(img_path):
    image_data = sitk.ReadImage(img_path)
    return image_data 


def read_nii_img(img_path):
    image_data = np.array(nib.load(img_path).get_fdata())
    return image_data