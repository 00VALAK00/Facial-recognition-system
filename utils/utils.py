import numpy
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, InterpolationMode, ToPILImage
import torchvision
import matplotlib.pyplot as plt
import cv2
class Preprocessor:

    def __init__(self):
        transform = Compose([
            ToPILImage(),
            Resize((250, 250), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
        self.preprocessor = transform

    def process(self, img):
        processed_img = self.preprocessor(img)
        return processed_img


def get_database_tensors(preprocessor: Preprocessor):

    iheb_img = torchvision.io.read_image("C:/Users/Iheb/Desktop/projects/facial-recognition-system/database/iheb.jpg",
                                          mode=torchvision.io.ImageReadMode.RGB)
    rayen_img = torchvision.io.read_image("C:/Users/Iheb/Desktop/projects/facial-recognition-system/database/rayen.jpg",mode=torchvision.io.ImageReadMode.RGB)
    abdo_img = torchvision.io.read_image("C:/Users/Iheb/Desktop/projects/facial-recognition-system/database/abdo.jpg",mode=torchvision.io.ImageReadMode.RGB)
    branda9=torchvision.io.read_image("C:/Users/Iheb/Desktop/projects/facial-recognition-system/database/branda9.jpg",mode=torchvision.io.ImageReadMode.RGB)
    rayen_img = preprocessor.process(rayen_img)
    abdo_img = preprocessor.process(abdo_img)
    iheb_img = preprocessor.process(iheb_img)
    branda9=preprocessor.process(branda9)
    database=np.array(["iheb","rayen","branda9","abdo_img"])
    encoding=torch.stack((iheb_img,rayen_img,branda9,abdo_img),dim=0)
    return database,encoding


def get_crop_transformer():

    transform = Compose([
        ToTensor(),
        Resize((250, 250), interpolation=InterpolationMode.BILINEAR),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform



def main():
    transform=Preprocessor()
    database,encodings=get_database_tensors(transform)

    print(encodings.shape)
    iheb=encodings[0].numpy().transpose(1,2,0)
    cv2.imshow(iheb)
    cv2.waitKey(12)

if __name__=="__main__":
    main()