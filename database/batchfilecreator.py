from pathlib import Path
from sys import platform
import click

thispath = Path(__file__).resolve()


def modify_transform_parameters(transform_parameters_directory, num_parameters):
    for x in range(num_parameters):
        transform_parameters_file = Path(transform_parameters_directory / f"TransformParameters.{str(x)}.txt")
        if not transform_parameters_file.exists():
            raise TypeError(f"File {transform_parameters_file} does "
                            f"not exist, make sure to have run elastix_"
                            f"{transform_parameters_directory.parent.parent.parent.parent.stem}_"
                            f"{transform_parameters_directory.parent.parent.parent.parent.parent.stem}_"
                            f"{transform_parameters_directory.parent.parent.stem}."
                            f"{'bat' if 'win32' in platform else 'sh'} correctly. ")
        else:
            with open(transform_parameters_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.find('FinalBSplineInterpolationOrder ') != -1:
                        lines[lines.index(line)] = '(FinalBSplineInterpolationOrder 0)\n'
                        break

            with open(transform_parameters_file,'w') as f:
                f.writelines(lines)


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
    Path(thispath.parent.parent / "Elastix_batch_files").mkdir(exist_ok=True, parents=True)
    with open(
            thispath.parent.parent / f"Elastix_batch_files/elastix_{parameter}_{directory_name}_{fixed_brain[0].parent.stem}."
                                     f"{'bat' if 'win32' in platform else 'sh'}", 'w') as f:
        f.write(f"ECHO Registration of the brains in training folder into the fixed image {fixed_brain[0].parent.stem} "
                f"with AFFINE+BSpline transformation \n\n")
        for train_file in images_files_train:
            output = Path(f"{fixed_brain[0].parent.parent.parent}/{directory_name}/{parameter}" \
                          f"/{fixed_brain[0].parent.parent.stem}/{fixed_brain[0].parent.stem}/brains/{train_file.parent.stem}")
            if "darwin" in platform:
                registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH \n\n" \
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
        f.write(f"ECHO End registration of training brains into IBSR_{number_file} \n")
        f.write("PAUSE")


def transformix_batch_file(number_file, parameter, bf_correction=False):
    datadir = thispath.parent.parent / "data"
    datadir_param = Path(thispath.parent.parent / Path("Elastix_batch_files") / parameter)
    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    number_parameters = len(parameters_files)
    images_files_train = [i for i in datadir.rglob("*seg.nii.gz") if "Training" in str(i)]
    fixed_brain = [i for i in datadir.rglob("*seg.nii.gz") if number_file in str(i)]
    if bf_correction:
        transform_param_dir = Path(thispath.parent.parent / f"data/BFC_registration/{parameter}"
                                                            f"/{fixed_brain[0].parent.parent.stem}" \
                                                            f"/{fixed_brain[0].parent.stem}/brains")
        directory_name = 'BFC_registration'
    else:
        transform_param_dir = Path(
            thispath.parent.parent / f"data/NoBFC_registration/{parameter}"
                                     f"/{fixed_brain[0].parent.parent.stem}" \
                                     f"/{fixed_brain[0].parent.stem}/brains")
        directory_name = 'NoBFC_registration'
    # Modifying the TransformParameters.x.txt files
    for train_file in images_files_train:
        modify_transform_parameters(Path(transform_param_dir / train_file.parent.stem), number_parameters)
    with open(
            thispath.parent.parent / f"Elastix_batch_files/transformix_{parameter}_{directory_name}_{fixed_brain[0].parent.stem}."
                                     f"{'bat' if 'win32' in platform else 'sh'}", 'w') as f:
        f.write("ECHO Transformix at work \n\n")
        for train_file in images_files_train:
            output = Path(f"{fixed_brain[0].parent.parent.parent}/{directory_name}/{parameter}"
                          f"/{fixed_brain[0].parent.parent.stem}/{fixed_brain[0].parent.stem}/labels/{train_file.parent.stem}")
            param = Path(f"{transform_param_dir}/{train_file.parent.stem}/TransformParameters.{number_parameters-1}.txt")
            if "darwin" in platform:
                registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH \n\n" \
                               f"mkdir -p {output} \n\n"
            elif "win32" in platform:
                registration = f"mkdir {output} \n\n"
            registration = f"{registration}" \
                           f"transformix -in {train_file}" \
                           f" -out {output}" \
                           f" -tp {param} \n\n"
            f.write(registration)
        f.write(f"ECHO End registration of training labels into IBSR_{number_file} \n")
        f.write("PAUSE")


@click.command()
@click.option(
    "--batch_type",
    default="elastix",
    help="Chose to create an elastix or transformix file"
)
@click.option(
    "--patient_number",
    help="Choose into which patient all the training brains are going to be registered to"
)
@click.option(
    "--bfc",
    default=False,
    help=
    "Whether to perform the "
)
@click.option(
    "--parameter",
    default="Par0010",
    help="name of the parameter folder; like Par0010, etc",
)
def main(batch_type, patient_number, bfc, parameter):

    if batch_type == 'elastix':
        elastix_batch_file(patient_number, parameter=parameter, bf_correction=bfc)
        print("Run the elastix file before creating the transformix file.")
    if batch_type == 'transformix':
        transformix_batch_file(patient_number, parameter=parameter, bf_correction=bfc)


if __name__ == "__main__":
    main()
