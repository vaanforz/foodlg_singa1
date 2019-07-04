# Foodlg

This repo shows how to train modern CNN models on food dataset for food classification. There are two phases: train and inference.

## Instructions

1. Go into `docker` folder, and run `bash build.sh` which will create a new docker image named foodlg-gpu,
 which includes basic environments (ubuntu 16.04 and basic tools) and necessary DL packages (tensorflow, keras and etc).

2. Prepare dataset, dataset can be the following: uec100, uec256, food101, food172, food191 and foodlg (a combination of datasets).
For dataset such as food191 uses some strange food names as the directory names, better to rename them based on the corresponding class indices file.
Split your whole dataset into train, validation and test folders inside its root folder. 
Count the number of images for train, validation and test, because we use flow_from_directory which will calculate the number of 
batches based on total number of images and batch size.

3. Go into `config` folder and create configurations for train and inference phases.

4. Go into `docker` folder and config the script (train.sh.example or inference.sh.example),
 which launches a docker container for training/inference.
You can also launch a docker container in python code by using `os.system('nvidia-docker ...')`.


