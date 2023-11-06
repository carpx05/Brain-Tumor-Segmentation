import os
import numpy as np
from utils import read_img_nii


def my_custom_generator_segmentation(data_paths, label_paths, batch_size):
    num_samples = len(data_paths)

    def generate_batch(batch_x, batch_y):
        items = []
        labels = []

        for item in batch_x:
            flair_img = read_img_nii(item[0])
            t1_img = read_img_nii(item[1])
            t2_img = read_img_nii(item[2])
            t1ce_img = read_img_nii(item[3])
            data = np.array([t1_img, t2_img, t1ce_img, flair_img], dtype=np.float32)
            items.append(data)

        for label_address in batch_y:
            # Simulated label data for demonstration
            label = np.random.randint(0, 5, size=(256, 256))
            ncr = label == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
            ed = label == 2  # Peritumoral Edema (ED)
            et = label == 4  # GD-enhancing Tumor (ET)
            y = np.array([ncr, ed, et], dtype=np.float32)
            labels.append(y)

        return np.array(items), np.array(labels)

    def data_generator():
        for i in range(0, num_samples, batch_size):
            batch_x = data_paths[i:i + batch_size]
            batch_y = label_paths[i:i + batch_size]
            yield generate_batch(batch_x, batch_y)

    return data_generator