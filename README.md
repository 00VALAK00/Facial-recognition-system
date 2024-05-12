# Facial Recognition Application With SNN & Yolov8

the following repo hightlights how to train a siamese NN on a image similiarty using a public available dataset (LFW) and Intergrating it with Yolov8-face.

## Workflow

1. Implementing the SNN architecture with DeepLearning and preparing the similar/different pairs of images.

2. Training & evaluating the model

3. Integrating the SNN with yolo 

## 1. Architecture 
VGG19 was used as encoder for the lack of data and some layers were added for the network to understand how to distinguish similar from different pairs
loss: contrastive loss
encoder: VGG19
batches: 16 batches
lr: 1e-5
device: GPU
duration 2hr
![image](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/befccd39-c850-437d-8248-a822f8c10ff9)

## 2. Performance 
![image](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/d2c2250a-d43d-43c5-b2ca-7a3104ebc01d)

![loss2](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/9ace3de8-5fff-4fbe-85d4-e2ce3a6b637e)

## Installation

1. Clone the repository.
2. Install the necessary dependencies.
3. Run the application

(Have fun trying it out)
