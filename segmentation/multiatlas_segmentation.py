from pathlib import Path
import click
import nibabel as nib
import numpy as np
from utils import dsc_score, read_groundtruth
from scipy.stats import mode
from tqdm import tqdm
from utils import csv_writer

thispath = Path.cwd().resolve()


def multiatlas_majority_voting_one(brain_patient, registration_folder, parameter_folder):

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

    output_dir = Path(thispath / "metrics")
    output_dir.mkdir(exist_ok=True, parents=True)
    header = ["Patient", "DICE WM", "DICE GM", "DICE CSF"]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voating_one_{brain_patient}.csv", "w", header)

    dice_per_tissue = dsc_score(final_segmentation, groundtruth[brain_patient])

    writer = [brain_patient, dice_per_tissue[0], dice_per_tissue[1], dice_per_tissue[2]]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voating_one_{brain_patient}.csv", "w", writer)

    print(f"\nDICE score {brain_patient}")
    print(f"WM: {dice_per_tissue[0]}")
    print(f"GM: {dice_per_tissue[1]}")
    print(f"CSF: {dice_per_tissue[2]}\n")


def multiatlas_majority_voting_all(registration_folder, parameter_folder):

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

    output_dir = Path(thispath / "metrics")
    output_dir.mkdir(exist_ok=True, parents=True)
    header = ["Patient", "DICE WM", "DICE GM", "DICE CSF"]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voating_all.csv", "w", header)

    for patient, labels in tqdm(zip(all_labels, all_labels.values()), desc=f"Segmenting brains", total=len(all_labels)):
        segmentation = mode(labels, axis=0).mode[0]
        dice_per_tissue = dsc_score(segmentation, groundtruth[patient])
        writer = [patient, dice_per_tissue[0], dice_per_tissue[1], dice_per_tissue[2]]
        csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voating_all.csv", "w", writer)

        print(f"\nDICE score {patient}")
        print(f"WM: {dice_per_tissue[0]}")
        print(f"GM: {dice_per_tissue[1]}")
        print(f"CSF: {dice_per_tissue[2]}\n")

        final_segmentations.append(segmentation)


@click.command()
@click.option(
    "--segmentation_option",
    default="majority_voting_all",
    help=
    "Choose the segmentation method. Among majority_voting_one, majority_voting_all",
)
@click.option(
    "--brain_patient",
    default="IBSR_11",
    help=
    "Name of the patient you want to segment. Only used in segmentation methods that you want yo segment only one"
    " patinet, not all of them at once.",
)
@click.option(
    "--registration_folder",
    default="NoBFC_registration",
    help=
    "Name of the folder with the registration results coming from elastix",
)
@click.option(
    "--parameter_folder",
    default="Par0010",
    help=
    "Name of the folder with the parameters used in the elastix registration for the registration folder selected",
)
def main(segmentation_option, brain_patient, registration_folder, parameter_folder):
    if segmentation_option == 'majority_voting_one':
        multiatlas_majority_voting_one(brain_patient, registration_folder, parameter_folder)

    elif segmentation_option == 'majority_voting_all':
        multiatlas_majority_voting_all(registration_folder, parameter_folder)


if __name__ == "__main__":
    main()
