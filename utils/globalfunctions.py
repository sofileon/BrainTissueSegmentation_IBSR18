import csv
from pathlib import Path
import numpy as np
import nibabel as nib

thispath = Path.cwd().resolve()


def csv_writer(file_path, name, action, data):
    """

    Parameters
    ----------
    file_path (Path from pathlib): path where to save the csv file
    name (string): csv name
    action (char): Either 'w' to write a new csv file or 'a' to append a new row
    data (list): Data to be appended to new row

    Returns
    -------
    """
    absolute_path = file_path / name
    with open(absolute_path, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


def read_groundtruth():
    datadir = Path(thispath / "data")
    validation_groundtruth = [i for i in datadir.rglob("*.nii.gz") if "Validation_Set" in str(i)
                              and "seg" in str(i)]
    groundtruth = {}
    for seg_path in validation_groundtruth:
        segmentation = nib.load(seg_path)
        segmentation = segmentation.get_fdata()[:, :, :, 0].astype(np.int8)
        groundtruth[seg_path.parent.stem] = segmentation

    return groundtruth
