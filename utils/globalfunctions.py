import csv
from pathlib import Path
import nibabel as nib
import numpy as np


def csv_writer(filename, action, row):
    """

    :param filename: string
    csv file name with its path
    :param action: char
     Either 'w' to write a new csv file or 'a' to append a new row
    :param row: list
     Data to be appended to new row

    :return:
    """
    filename = Path(filename)
    with open(filename, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()


def metadata_creation():
    """

    :return:
    """
    datadir = Path("data")
    images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i) and "seg" not in str(i)]
    images_files_validation = [i for i in datadir.rglob("*.nii.gz") if "Validation" in str(i) and "seg" not in str(i)]
    images_files_test = [i for i in datadir.rglob("*.nii.gz") if "Test" in str(i) and "seg" not in str(i)]
    header = ['Name', 'Dataset', 'Original Size', 'Voxel Size (xyz)',
              'DataType','Minimum intensity (non-zero)', 'Maximum intensity']
    csv_writer('data/Metadata_IBSR18.csv', 'w', header)
    for file in images_files_train:
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, 'Train', brain_image.shape[:3],
               tuple(np.diag(brain_image.affine)[:-1]), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer('data/Metadata_IBSR18.csv', 'a', row)

    for file in images_files_validation:
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, 'Validation', brain_image.shape[:3],
               tuple(np.diag(brain_image.affine)[:-1]), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer('data/Metadata_IBSR18.csv', 'a', row)

    for file in images_files_test:
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, 'Test', brain_image.shape[:3],
               tuple(np.diag(brain_image.affine)[:-1]), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer('data/Metadata_IBSR18.csv', 'a', row)
    print('Metadata csv created at data/Metadata_IBSR18.csv')
