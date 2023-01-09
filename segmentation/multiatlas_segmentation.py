from pathlib import Path
import click
import nibabel as nib
import numpy as np
from utils import metrics_4_segmentation, read_groundtruth
from scipy.stats import mode
from tqdm import tqdm
from utils import csv_writer
import time
import datetime

thispath = Path.cwd().resolve()


def multiatlas_majority_voting_one(brain_patient, registration_folder, parameter_folder):
    start = time.time()
    datadir = thispath / "data"

    registered_image_for_one = [i for i in datadir.rglob("*.nii.gz") if registration_folder in str(i)
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
    header = ["Patient", "DICE WM", "DICE GM", "DICE CSF", "HD WM", "HD GM", "HD CSF", "RAVD WM", "RAVD GM", "RAVD CSF"]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voting_one_{brain_patient}.csv", "w", header)

    dice_per_tissue, hd_per_tissue, ravd_per_tissue = metrics_4_segmentation(final_segmentation,
                                                                             groundtruth[brain_patient])

    writer = [brain_patient, dice_per_tissue[0], dice_per_tissue[1], dice_per_tissue[2],
              hd_per_tissue[0], hd_per_tissue[1], hd_per_tissue[2],
              ravd_per_tissue[0], ravd_per_tissue[1], ravd_per_tissue[2]]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voting_one_{brain_patient}.csv", "a", writer)

    final_time = time.time()-start
    writer = ["Time", "{:0>8}".format(str(datetime.timedelta(seconds=final_time)))]
    csv_writer(output_dir, f"{registration_folder}_multiAtlas_majority_voting_one_{brain_patient}.csv", "a", writer)

    print(f"DICE, HD, RAVD computed and saved in a"
          f" {registration_folder}_multiAtlas_majority_voting_one_{brain_patient}.csv in {output_dir}")


def multiatlas_majority_voting_all(registration_folder, parameter_folder, test=False):
    start = time.time()
    datadir = thispath / "data"

    registered_images_train = [i for i in datadir.rglob("*.nii.gz") if registration_folder in str(i)
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
            labels.append(label)

        all_labels[registered_image.parent.parent.parent.stem] = labels

    output_dir = Path(thispath / "metrics")
    if not test:
        groundtruth = read_groundtruth()
        output_dir.mkdir(exist_ok=True, parents=True)
        header = ["Patient", "DICE WM", "DICE GM", "DICE CSF", "HD WM", "HD GM", "HD CSF", "RAVD WM", "RAVD GM",
                  "RAVD CSF"]
        csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                               f"_multiAtlas_majority_voting_all.csv", "w", header)

    for patient, labels in tqdm(zip(all_labels, all_labels.values()), desc=f"Segmenting brains", total=len(all_labels)):
        affine = labels[0].affine
        labels = [label.get_fdata().astype(np.int8) for label in labels]
        segmentation = mode(labels, axis=0).mode[0]

        if test:
            seg_header = nib.Nifti1Header()
            segmentation_write = nib.Nifti1Image(segmentation, affine, seg_header)
            output_segdir = Path(thispath / "Final_segmentations" / registration_folder)
            output_segdir.mkdir(exist_ok=True, parents=True)
            nib.save(segmentation_write, f'{str(output_segdir)}/{patient}_segmentation.nii.gz')
        else:
            dice_per_tissue, hd_per_tissue, ravd_per_tissue = metrics_4_segmentation(segmentation,
                                                                                     groundtruth[patient])
            writer = [patient, dice_per_tissue[0], dice_per_tissue[1], dice_per_tissue[2],
                      hd_per_tissue[0], hd_per_tissue[1], hd_per_tissue[2],
                      ravd_per_tissue[0], ravd_per_tissue[1], ravd_per_tissue[2]]

            csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                                   f"_multiAtlas_majority_voting_all.csv", "a", writer)

    final_time = time.time()-start
    writer = ["Time", "{:0>8}".format(str(datetime.timedelta(seconds=final_time)))]
    csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                           f"_multiAtlas_majority_voting_all.csv", "a", writer)

    print(f"DICE, HD, RAVD computed and saved in a {registration_folder}_{parameter_folder}"
          f"_multiAtlas_majority_voating_all.csv in {output_dir}")


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
    default="BFC_registration",
    help=
    "Name of the folder with the registration results coming from elastix",
)
@click.option(
    "--parameter_folder",
    default="Par0010",
    help=
    "Name of the folder with the parameters used in the elastix registration for the registration folder selected",
)
@click.option(
    "--test_boolean",
    default=False,
    help=
    "To save the final segmentations for the Test set if True",
)
def main(segmentation_option, brain_patient, registration_folder, parameter_folder, test_boolean):
    if segmentation_option == 'majority_voting_one':
        multiatlas_majority_voting_one(brain_patient, registration_folder, parameter_folder)

    elif segmentation_option == 'majority_voting_all':
        multiatlas_majority_voting_all(registration_folder, parameter_folder, test_boolean)


if __name__ == "__main__":
    main()
