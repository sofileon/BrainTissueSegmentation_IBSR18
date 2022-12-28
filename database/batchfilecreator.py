from pathlib import Path
from sys import platform

thispath = Path(__file__).resolve()


def elastix_batch_file(number_file, parameter, bf_correction=False):
    datadir = thispath.parent.parent / "data"
    datadir_param = Path(thispath.parent.parent / Path("Elastix_batch_files") / parameter)
    if bf_correction:
        images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i)
                              and "seg" not in str(i)
                              and "BFC" in str(i)
                              and 'registration' not in str(i)]
        fixed_brain = [i for i in datadir.rglob("*.nii.gz") if "seg" not in str(i)
                       and "BFC" in str(i)
                       and number_file in str(i)
                       and 'registration' not in str(i)]
        directory_name = 'BFC_registration'
    else:
        images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i)
                              and "seg" not in str(i)
                              and "BFC" not in str(i)]
        fixed_brain = [i for i in datadir.rglob("*.nii.gz") if "seg" not in str(i)
                           and "BFC" not in str(i)
                           and number_file in str(i)]
        directory_name = 'NoBFC_registration'

    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    parameters_files = sorted(parameters_files, key=lambda i: str(i.stem))
    Path(thispath.parent.parent/"Elastix_batch_files").mkdir(exist_ok=True, parents=True)
    with open(thispath.parent.parent/f"Elastix_batch_files/elastix_{parameter}_{directory_name}_{fixed_brain[0].parent.stem}."
                                     f"{'bat' if 'win32' in platform else 'sh'}", 'w') as f:
        f.write(f"ECHO Registration of the brains in training folder into the fixed image {fixed_brain[0].parent.stem} "
                f"with AFFINE+BSpline transformation \n\n")
        for train_file in images_files_train:
            output = Path(f"{fixed_brain[0].parent.parent.parent}/{directory_name}" \
                    f"/{fixed_brain[0].parent.parent.stem}/{fixed_brain[0].parent.stem}/brains/{train_file.parent.stem}")
            if "darwin" in platform:
                registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH \n\n"\
                                        f"mkdir -p {output} \n\n"
            elif "win32" in platform:
                registration = f"mkdir {output} \n\n"
            registration = f"{registration}" \
                           f"elastix -f {fixed_brain[0]}" \
                           f" -m {train_file}" \
                           f" -out {output}"
            for param in parameters_files:
                registration = f"{registration} -p {param}"
            registration = f"{registration} \n\n"
            f.write(registration)


def transformix_batch_file(number_file, parameter, bf_correction=False):
    datadir = thispath.parent.parent / "data"
    datadir_param = Path(thispath.parent.parent / Path("Elastix_batch_files") / parameter)
    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    number_parameters = len(parameters_files)-1
    images_files_train = [i for i in datadir.rglob("*seg.nii.gz") if "Training" in str(i)]
    fixed_brain = [i for i in datadir.rglob("*seg.nii.gz") if number_file in str(i)]
    if bf_correction:
        transform_param_dir = Path(thispath.parent.parent / f"data/BFC_registration/{fixed_brain[0].parent.parent.stem}" \
                                                       f"/{fixed_brain[0].parent.stem}/brains")
        directory_name = 'BFC_registration'
    else:
        transform_param_dir = Path(thispath.parent.parent / f"data/NoBFC_registration/{fixed_brain[0].parent.parent.stem}" \
                                                       f"/{fixed_brain[0].parent.stem}/brains")
        directory_name = 'NoBFC_registration'
    with open(thispath.parent.parent/f"Elastix_batch_files/transformix_{parameter}_{directory_name}_{fixed_brain[0].parent.stem}."
                                     f"{'bat' if 'win32' in platform else 'sh'}", 'w') as f:
        f.write("ECHO Transformix at work \n\n")
        for train_file in images_files_train:
            output = Path(f"{fixed_brain[0].parent.parent.parent}/{directory_name}"
                          f"/{fixed_brain[0].parent.parent.stem}/{fixed_brain[0].parent.stem}/labels/{train_file.parent.stem}")
            param = Path(f"{transform_param_dir}/{train_file.parent.stem}/TransformParameters.{number_parameters}.txt")
            if "darwin" in platform:
                registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH \n\n"\
                                        f"mkdir -p {output} \n\n"
            elif "win32" in platform:
                registration = f"mkdir {output} \n\n"
            registration = f"{registration}"\
                           f"transformix -in {train_file}" \
                           f" -out {output}" \
                           f" -tp {param} \n\n"
            f.write(registration)


if __name__ == "__main__":
    elastix_batch_file('17',parameter='Par0010',bf_correction=True)

    transformix_batch_file('17', parameter='Par0010', bf_correction=True)

