from pathlib import Path
import click
import nibabel as nib
import numpy as np
from utils import dsc_score, read_groundtruth
from scipy.stats import mode
from tqdm import tqdm

thispath = Path.cwd().resolve()


def segment_all_majority_voting(registration_folder, parameter_folder):
    # registration_folder = "NoBFC_registration"
    # parameter_folder = "Par0010"

    datadir = thispath / "data"

    registered_images_train = [i for i in datadir.rglob("*.nii") if registration_folder in str(i)
                          and parameter_folder
                          and "labels" in str(i)]

    registered_images_for_all = [
        registered_images_train[i: i+10] for i in range(0, len(registered_images_train), 10)
    ]

    all_labels = {}
    for registered_image_for_one in registered_images_for_all:
        labels = []
        for registered_image in registered_image_for_one:
            label = nib.load(registered_image)
            label = label.get_fdata().astype(np.int8)
            labels.append(label)

        labels = np.asarray(labels)
        all_labels[registered_image.parent.parent.parent.stem] = labels

    groundtruth = read_groundtruth()
    final_segmentations = []

    for i, labels in tqdm(enumerate(all_labels.values())):
        segmentation = mode(labels, axis=0).mode[0]
        final_segmentations.append(segmentation)
        dsc_score(final_segmentations[i], groundtruth[i])


def segment_one_majority_voting(brain_patient, registration_folder, parameter_folder):
    # registration_folder = "NoBFC_registration"
    # parameter_folder = "Par0010"
    # brain_patient = "IBSR_11"

    datadir = thispath / "data"

    registered_image_for_one = [i for i in datadir.rglob("*.nii") if registration_folder in str(i)
                                and parameter_folder
                                and brain_patient in str(i)
                                and "labels" in str(i)]

    labels = []
    for registered_image in registered_image_for_one:
        label = nib.load(registered_image)
        label = label.get_fdata().astype(np.int8)
        labels.append(label)

    labels = np.asarray(labels)

    final_segmentation = mode(labels, axis=0).mode[0]

    groundtruth = read_groundtruth()
    dsc_score(final_segmentation, groundtruth[brain_patient])
    print(groundtruth)
