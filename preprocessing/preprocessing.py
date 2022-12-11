from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import cv2 as cv
import numpy as np
thispath = Path(__file__).resolve()


def BiasFieldCorrection(dataset_choice):
    if dataset_choice != 'Test' and dataset_choice != 'Validation' and dataset_choice != 'Training':
        raise TypeError("Not a valid input argument, enter either 'Training', "
                        "'Validation' or 'Test'")

    datadir = thispath.parent.parent/'data'
    metadata = pd.read_csv(datadir/'Metadata_IBSR18.csv', header=0)
    brains = metadata['Name'].loc[metadata['Dataset'] == f'{dataset_choice}_Set'].tolist()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    corrector.SetMaximumNumberOfIterations([20] * numberFittingLevels)

    for i, brain in zip(tqdm(range(len(brains)), desc=f'{dataset_choice} Brains Bias Field Correction: '), brains):
        input_brain = sitk.ReadImage(datadir/f'{dataset_choice}_Set/{brain}/{brain}.nii.gz', sitk.sitkFloat32)
        maskImage = sitk.OtsuThreshold(input_brain, 0, 1)
        corrected_image = corrector.Execute(input_brain, maskImage)
        sitk.WriteImage(
            corrected_image, datadir/f"{dataset_choice}_Set/{brain}/BFC_{brain}.nii.gz"
        )


if __name__ == "__main__":
    BiasFieldCorrection('Validation')

