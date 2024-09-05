# yolov4matlab

## Requirements
- MATLAB 2024a with Add-ons Computer Vision Toolbox and Deep Learning Toolbox installed
- Python 3
- YOLOv8 txt format dataset

## Usage
First you must generate the .mat table containing the dataset information, specifying the dataset path:
```bash
python dataset_to_mat.py "<dataset_path>"
```
Then, open the rede_copel.m file and change line 19 with the dataset path, maining the rede_copel.m file:
```
19 data = load("<dataset_path>\rede_copel.mat");
```
You can now run the rede_copel.m script inside MATLAB and it should train the model

**OBS:** You can also change the model parameters for training in the rede_copel.m

**OBS 2:** If you change the dataset path after creating the .mat file, you should create it again as the matlab script wont be able to find the images
