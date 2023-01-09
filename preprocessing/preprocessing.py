from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage import exposure

thispath = Path(__file__).resolve()


def bias_field_correction(dataset_choice):
    if dataset_choice != 'Test' and dataset_choice != 'Validation' and dataset_choice != 'Training':
        raise TypeError("Not a valid input argument, enter either 'Training', "
                        "'Validation' or 'Test'")

    datadir = thispath.parent.parent/'data'
    metadata = pd.read_csv(datadir/'Metadata_IBSR18.csv', header=0)
    brains = metadata['Name'].loc[metadata['Dataset'] == f'{dataset_choice}_Set'].tolist()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    number_fitting_levels = 4
    corrector.SetMaximumNumberOfIterations([20] * number_fitting_levels)

    for i, brain in zip(tqdm(range(len(brains)), desc=f'{dataset_choice} Brains Bias Field Correction: '), brains):
        input_brain = sitk.ReadImage(datadir/f'{dataset_choice}_Set/{brain}/{brain}.nii.gz', sitk.sitkFloat32)
        mask_image = sitk.OtsuThreshold(input_brain, 0, 1)
        corrected_image = corrector.Execute(input_brain, mask_image)
        sitk.WriteImage(
            corrected_image, datadir/f"{dataset_choice}_Set/{brain}/BFC_{brain}.nii.gz"
        )


def normalization(dataset_choice):

    datadir = thispath.parent.parent / 'data'
    metadata = pd.read_csv(datadir / 'Metadata_IBSR18.csv', header=0)
    brains = metadata['Name'].loc[metadata['Dataset'] == f'{dataset_choice}_Set'].tolist()

    for i, brain in zip(tqdm(range(len(brains)), desc=f'{dataset_choice} Normalization: '), brains):
        input_brain_sitk = sitk.ReadImage(datadir/f'{dataset_choice}_Set/{brain}/BFC_{brain}.nii.gz', sitk.sitkFloat32)
        input_brain = sitk.GetArrayFromImage(input_brain_sitk)
        normalized_brain = exposure.rescale_intensity(input_brain,
                                                      in_range="image",
                                                      out_range=(0, 255))
        normalized_brain = sitk.GetImageFromArray(normalized_brain)
        normalized_brain.CopyInformation(input_brain_sitk)
        sitk.WriteImage(
            normalized_brain, datadir/f"{dataset_choice}_Set/{brain}/Normalized_{brain}.nii.gz"
        )


if __name__ == "__main__":
    bias_field_correction('Training')
    bias_field_correction('Validation')
    bias_field_correction('Test')
    normalization('Training')
    normalization('Validation')
    normalization('Test')
