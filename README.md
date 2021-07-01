# Train cats vs dogs instance segmentation model
## 1. Introduction
Train a cats vs dogs instance segmentation model from scratch using Mask R-CNN in PyTorch

In this project we use the popular Oxford-IIIT Pet Dataset

## 2 Prerequisites
- Python 3
- Numpy
- PIL
- PyTorch 1.8.1
- Torchvision
- Opencv (cv2)

## 3. Installation

1. Clone the respository
```bash
git clone https://github.com/harshatejas/cats_vs_dogs_instance_segmentation.git
cd cats_vs_dogs_instance_segmentation/
```
2. Dowload the dataset

   Head over to https://www.robots.ox.ac.uk/~vgg/data/pets/ to download The Oxford-IIIT Pet Dataset
   
   This will download and extract the images and annotations to cats_vs_dogs_instance_segmentation directory
   
```bash
# Download the dataset
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```
```bash
# Extract the dataset
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

   3. The dataset is split into train and validation by train.py, it is split based on the availability of annotations (.xml)
```bash
OxfordDataset/
 Images/
   [xxx].jpg
   ...
 Images_val/
   [xxx].jpg
   ...
 Masks/
   [xxx].png
   ...
 Masks_val/
   [xxx].png
   ...  
 Xmls/
   [xxx].xml
   ...
``` 

## 4. Train
Modify Hyperparameters in train.py
```bash
train.py
```

## 5. Test
predict.py is designed to run predictions on the images in validation folder (Images_val)

Change the filename and saved_model in predict.py
```bash
predict.py
```

## 6. Predicted Images
Here are some sample output images predicted by saved_model/model
