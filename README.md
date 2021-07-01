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
