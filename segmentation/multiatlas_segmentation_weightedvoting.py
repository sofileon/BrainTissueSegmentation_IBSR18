import numpy as np
import nibabel as nib
import pandas as pd
import click
from utils import metrics_4_segmentation, read_groundtruth, csv_writer
from tqdm import tqdm
import time
import datetime
from pathlib import Path

thispath = Path(__file__).resolve()


def multiatlas_weighted_majority_voting_all(registration_folder, parameter_folder, test=False):
    start = time.time()
    datadir = thispath.parent.parent / "data"
    registered_images_train = [i for i in datadir.rglob("*.nii.gz") if registration_folder in str(i)
                               and parameter_folder
                               and "labels" in str(i)]
    registered_images_for_all = [
        registered_images_train[i: i + 10] for i in range(0, len(registered_images_train), 10)
    ]
    all_labels = {}
    for registered_image_for_one in registered_images_for_all:
        labels = [nib.load(registered_image) for registered_image in registered_image_for_one]
        all_labels[registered_image_for_one[0].parent.parent.parent.stem] = labels

    output_dir = Path(thispath.parent.parent / "metrics")
    if not test:
        groundtruth = read_groundtruth()
        output_dir.mkdir(exist_ok=True, parents=True)
        header = ["Patient", "DICE WM", "DICE GM", "DICE CSF", "HD WM", "HD GM", "HD CSF", "RAVD WM", "RAVD GM", "RAVD CSF"]
        csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                               f"_multiAtlas_weighted_majority_voting_all.csv", "w", header)

    for patient, labels in tqdm(all_labels.items(), desc=f"Segmenting brains", total=len(all_labels)):
        # labels of size 10*256*128*256
        affine = labels[0].affine
        labels = [label.get_fdata().astype(np.int8) for label in labels]
        labels = np.asarray(labels)
        white_matter = labels.copy()
        white_matter[labels != 3] = 0.
        white_matter[labels == 3] = 1.
        gray_matter = labels.copy()
        gray_matter[labels != 2] = 0.
        gray_matter[labels == 2] = 1.
        csf = labels.copy()
        csf[labels != 1] = 0.
        # Assigning weights
        metadata = pd.read_csv(datadir / 'Metadata_IBSR18.csv', index_col=0)
        patient_voxel_size = metadata['Voxel Size (xyz)'].loc[patient].replace('(', '').replace(')', '')
        patient_voxel_size = np.fromstring(patient_voxel_size, dtype=float, sep=',')
        train_voxel_size = metadata['Voxel Size (xyz)'].drop(index=patient)
        train_voxel_size = train_voxel_size[metadata.Dataset == 'Training_Set']
        train_voxel_size = np.array(
            [np.fromstring(x.replace('(', '').replace(')', ''), dtype=float, sep=',') for x in train_voxel_size])
        dst = np.linalg.norm(train_voxel_size - patient_voxel_size, axis=1)
        w = np.zeros([10])
        w[np.where(dst == np.amax(dst))] = .5
        w[np.where(dst == 0)] = 4
        w[np.where(w == 0)] = 1
        w = w.reshape([10, 1, 1, 1])

        # Doing the majority weighted voting
        label_3 = white_matter * w
        label_2 = gray_matter * w
        label_1 = csf * w

        weighted_labels = [label_1, label_2, label_3]

        for i, label in enumerate(weighted_labels):
            sum_label = np.sum(label, axis=0)
            if i == 0:
                voting = sum_label.flatten()
            else:
                voting = np.vstack([voting, sum_label.flatten()])
        segmentation = np.argmax(voting, axis=0)
        for i in range(voting.shape[1]):
            if voting[segmentation[i], i] == 0:
                segmentation[i] = -1
        segmentation = segmentation + 1
        segmentation = segmentation.reshape(labels[0].shape)

        if test:
            output_segdir = Path(thispath.parent.parent / "Final_segmentations" / registration_folder)
            output_segdir.mkdir(exist_ok=True, parents=True)
            seg_header = nib.Nifti1Header()
            seg_im = nib.Nifti1Image(segmentation, affine, seg_header)
            nib.save(seg_im,  f'{str(output_segdir)}/{patient}_segmentation.nii.gz')
        else:
            dice_per_tissue, hd_per_tissue, ravd_per_tissue = metrics_4_segmentation(segmentation,
                                                                                     groundtruth[patient])
            writer = [patient, dice_per_tissue[0], dice_per_tissue[1], dice_per_tissue[2],
                      hd_per_tissue[0], hd_per_tissue[1], hd_per_tissue[2],
                      ravd_per_tissue[0], ravd_per_tissue[1], ravd_per_tissue[2]]
            csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                                   f"_multiAtlas_weighted_majority_voting_all.csv", "a", writer)
    final_time = time.time() - start
    writer = ["Time", "{:0>8}".format(str(datetime.timedelta(seconds=final_time)))]
    csv_writer(output_dir, f"{registration_folder}_{parameter_folder}"
                           f"_multiAtlas_weighted_majority_voting_all.csv", "a", writer)


@click.command()
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
def main(registration_folder, parameter_folder, test_boolean):
    multiatlas_weighted_majority_voting_all(registration_folder, parameter_folder, test=test_boolean)


if __name__ == "__main__":
    main()
