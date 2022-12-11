from pathlib import Path

thispath = Path(__file__).resolve()


def batch_file(number_file, bf_correction=False):
    datadir = thispath.parent.parent / "data"
    if bf_correction:
        images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i)
                              and "seg" not in str(i)
                              and "BFC" in str(i)]
        fixed_brain = [i for i in datadir.rglob("*.nii.gz") if "seg" not in str(i)
                           and "BFC" in str(i)
                           and number_file in str(i)]
        directoryname = 'BFC_registration'
    else:
        images_files_train = [i for i in datadir.rglob("*.nii.gz") if "Training" in str(i)
                              and "seg" not in str(i)
                              and "BFC" not in str(i)]
        fixed_brain = [i for i in datadir.rglob("*.nii.gz") if "seg" not in str(i)
                           and "BFC" not in str(i)
                           and number_file in str(i)]
        directoryname = 'NoBFC_registration'

    with open(thispath.parent.parent/f'Elastix_batch_files/{directoryname}_{fixed_brain[0].parent.stem}.bat', 'w') as f:
        f.write("ECHO Registration of the folder into the fixed image IBSR_11.nii with AFFINE+BSpline transformation \n\n")
        for train_file in images_files_train:
            registration = f"mkdir {fixed_brain[0].parent}\{directoryname}\{train_file.parent.stem} \n\n" \
                           f"elastix -f {fixed_brain[0]}" \
                           f" -m {train_file}" \
                           f" -out {fixed_brain[0].parent}\{directoryname}\{train_file.parent.stem}" \
                           f" -p {thispath.parent.parent}\Elastix_batch_files\Par0010\Par0010affine.txt " \
                           f" -p {thispath.parent.parent}\Elastix_batch_files\Par0010\Par0010bspline.txt \n\n"
            f.write(registration)


if __name__ == "__main__":
    batch_file('17')

