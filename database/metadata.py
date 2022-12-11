from pathlib import Path
thispath = Path(__file__).resolve()
import nibabel as nib
import numpy as np
from utils import csv_writer
from tqdm import tqdm


def metadata_creation():
    """

    :return:
    """
    datadir = thispath.parent.parent/"data"
    images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i) and "seg" not in str(i)]
    images_files_validation = [i for i in datadir.rglob("*.nii.gz") if "Validation" in str(i) and "seg" not in str(i)]
    images_files_test = [i for i in datadir.rglob("*.nii.gz") if "Test" in str(i) and "seg" not in str(i)]
    header = ['Name', 'Dataset', 'Original Size', 'Voxel Size (xyz)',
              'DataType','Minimum intensity (non-zero)', 'Maximum intensity']
    csv_writer(datadir, 'Metadata_IBSR18.csv', 'w', header)
    for i, file in zip(tqdm(range(len(images_files_train)), desc='Train files'), images_files_train):
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, file.parent.parent.stem, brain_image.shape[:3],
               tuple(np.around(np.diag(brain_image.affine)[:-1], decimals=3)), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer(datadir, 'Metadata_IBSR18.csv', 'a', row)

    for i, file in zip(tqdm(range(len(images_files_validation)), desc='Validation files'), images_files_validation):
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, file.parent.parent.stem, brain_image.shape[:3],
               tuple(np.around(np.diag(brain_image.affine)[:-1], decimals=3)), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer(datadir, 'Metadata_IBSR18.csv', 'a', row)

    for i, file in zip(tqdm(range(len(images_files_test)), desc='Test files'), images_files_test):
        brain_image = nib.load(file)
        brain_data = np.squeeze(brain_image.get_fdata())
        min_value = np.min(brain_data[brain_data.nonzero()])
        max_value = np.max(brain_data)
        row = [Path(file.stem).stem, file.parent.parent.stem, brain_image.shape[:3],
               tuple(np.around(np.diag(brain_image.affine)[:-1], decimals=3)), brain_image.get_data_dtype(),
               min_value, max_value]
        # Notice that to remove both sufixxes, .nii.gz, a double stem was applied
        csv_writer(datadir, 'Metadata_IBSR18.csv', 'a', row)
    print(f'Metadata csv created at {datadir}')
