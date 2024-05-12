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
Dataset samples:17000
batches: 16 batches
lr: 1e-5
device: GPU
duration 2hr
contrastive loss margin parameter set to : 2
![image](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/befccd39-c850-437d-8248-a822f8c10ff9)

## 2. Performance 
![accuracy2](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/8f120649-a6ed-4c3b-b283-96a33030e0bf)

![loss2](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/9ace3de8-5fff-4fbe-85d4-e2ce3a6b637e)

## 3. Test
Two images are similar if the result array is under the margin threshold else similar
![image](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/a6c26c5f-4abc-4203-93d1-0ac68d394cb3)
![image](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/a66e947c-eeae-4227-970a-481ec5bcfaa5)

## 4. Integrating it with Yolo 
- Database of Encoded Images:  the database contains encoded representations of faces. These encoded representations are essentially numerical vectors derived from the facial features of each person.

- Comparison with Database: To identify a given face, you compare its encoded representation with those stored in the database. This comparison involves measuring the similarity  between the encoded representation of the face in frame and each encoded representation in the database.
this is done in parallel using torch data structure.

- Finding the Closest Match: After measuring the distance between the encoded representation of the face in question and each encoded representation in the database, you identify the one with the lowest score. This lowest score corresponds to the closest match in the database.


## 5. Results 
to try it out. the script uses a video and performs the detection on every frame detecing and annotating the person identity 
![8pu0lo](https://github.com/00VALAK00/Facial-recognition-system/assets/117487025/5a87d117-dbc5-4dc9-bd4e-fc7d68f60239)
