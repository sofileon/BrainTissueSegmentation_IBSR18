# BrainTissueSegmentation_IBSR18
This repository contains the code for Brain image segmentation of White Matter (WM), Grey matter (GM) and 
Cerebrospinal fluid (CSF) in the IBSR_18 dataset. 

## Setting up the environment
- Create a conda environment
```
conda create -n ctreg python==3.9.13 anaconda -y && conda activate ctreg
```
- Install the requirements
```
pip install -r requirements.txt
```

## Metadata creation

First we create the metadata of the IBSR_18 dataset necessary to run the algorithm.Run the following line to perform
this step.
```
python main.py 
```

## Preprocessing
Preprocessing of the dataset that consists in Bias Field correction and Normalization between 0 and 255.
Run the following line to perform this step.
```
python -m preprocessing.preprocessing
```

## Registration
Registration of the training dataset into Validation or Test to build the multi-atlases.

### Elastix 
Run the `batchfilecreator.py` to create the elastix batch files to register all the brains into the Validation or Test
sets. For example, run the following line of code for Validation.
```
python -m  database.batchfilecreator --batch_type elastix -set_option Validation --parameter Par0009 
```
That outputs 5 batch files that you should run before going to the next section in a folder that is created in 
`cwd()\elastix\NEW_FOLDER\elastix`

### Transformix 
Run the `batchfilecreator.py` to create the elastix batch files to run transformix and register all the labels into
the Validation or Test sets. For example, run the following line of code for Validation.
```
python -m  database.batchfilecreator --batch_type transformix -set_option Validation --parameter Par0009 
```
That outputs 5 batch files that you should run before going to the next section and perform the final segmentation, the
batch files are located in a folder that is created in `cwd()\elastix\NEW_FOLDER\transformix`

All the results coming from elastix are saved in a folder located in `cwd()\data\BFC_registration\Par0009`

## Segmentation 
To perform the final segmentation from the multi-atlas two strategies are proposed.
### Majority voting
Run the following line of code to perform the segmentation by using hard majority voting on the Validation.
```
python -m  segmentation.multiatlas_segmentation --parameter_folder Par0009 --registration_folder BFC_registration

```
That will output a *.csv file located in `cwd()\metrics` with the 3 different evaluation metrics per tissue DSC,
HD and RAVD.

Run the following line of code to perform the segmentation by using hard majority voting on the Test if all the previous
are run to build the multi-atlas for the Test set from the Training set.
```
python -m  segmentation.multiatlas_segmentation --parameter_folder Par0009 --test_boolean True
 --registration_folder BFC_registration
```
That will output *.nii.gz files with the final segmentations per patient in a folder call `cwd()\Final_segmentations`

### Majority voting weighted by voxel spacing similarity
Run the following line of code to perform the segmentation by using weighted majority voting on the Validation.
```
python -m  segmentation.multiatlas_segmentation --parameter_folder Par0009 --registration_folder BFC_registration

```
That will output a *.csv file located in `cwd()\metrics` with the 3 different evaluation metrics per tissue DSC,
HD and RAVD.

Run the following line of code to perform the segmentation by using hard majority voting on the Test if all the previous
are run to build the multi-atlas for the Test set from the Training set.
```
python -m  segmentation.multiatlas_segmentation --parameter_folder Par0009 --test_boolean True
 --registration_folder BFC_registration
```
That will output *.nii.gz files with the final segmentations per patient in a folder call `cwd()\Final_segmentations`
