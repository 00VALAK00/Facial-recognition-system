import cv2
import numpy as np
import torch
import torchvision.io
from ultralytics import YOLO
from pathlib import Path
import os
from model_class import VGG_SNN
import sys
from ultralytics.utils.plotting import Annotator, Colors
from utils.utils import Preprocessor, get_database_tensors, get_crop_transformer

main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("facial-recognition-system/model_class")
from torchvision.transforms import ToTensor
import cv2


class object_detector:

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"using device {self.device}")
        self.model = self.load_yolo_model()
        self.snn = self.load_face_recognition()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter('iheb.avi', fourcc, 20, (720, 1280))
        self.processor = Preprocessor()
        self.database, self.encodings = get_database_tensors(self.processor)
        self.crop_processor = get_crop_transformer()
        self.face_detector=cv2.CascadeClassifier("face.xml")

    def load_yolo_model(self):
        model = YOLO("yolov8m-face.pt")
        model.fuse()
        return model

    def predict(self, frame):
        return self.model(frame)

    def load_face_recognition(self):
        path = "C:/Users/Iheb/Desktop/projects/facial-recognition-system/vgg_snn10"
        model = torch.load(path)
        return model

    def retrieve_persons_locations(self, frame, results):
        for result in results:
            boxes = result.boxes
            # extract persons only
            persons = np.argwhere(boxes.cls.cpu().numpy() == 0)
            # get the bounding box for the persons only
            persons_locations = boxes.xyxy[persons]
            return persons_locations

    def plot_bbox(self, frame, persons):
        if persons.numel() > 0:
            print(persons[0][0])
            for person_location in persons[0]:
                x1, y1, x2, y2 = person_location[:4].cpu().numpy().astype(int)
                # Draw bbox for persons detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)
                person = ToTensor()((frame[y1:y2, x1:x2])).to(self.device)
                print(f"person type {type(person)}, shape {person.shape}")
                person = self.processor.process(person)
                print(f"person type {type(person)}, shape {person.shape}")
                person = person.repeat(self.database.shape[0], 1, 1, 1)
                print(f"repeated array shape {person.shape}")
                d, enc1, enc2 = self.snn(person, self.encodings)
                person_id = torch.argmin(d)

                text = self.database[person_id]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                text_thickness = 4
                text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
                text_width, text_height = text_size
                text_x = int((x1 + x2 - text_width) / 2)
                text_y = y1 - 10
                bg_x1 = text_x - 5
                bg_y1 = text_y - text_height - 5
                bg_x2 = text_x + text_width + 5
                bg_y2 = text_y + baseline + 5

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), cv2.FILLED)

                cv2.putText(frame, text=text, org=(text_x, text_y), fontFace=font, fontScale=font_scale,
                            color=(255, 255, 255), thickness=text_thickness)
                cv2.imshow("frame", frame)
            else:
                pass
            return frame

    def __call__(self, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            persons = self.retrieve_persons_locations(frame, results)
            annotated_frame = self.plot_bbox(frame, persons)
            self.output_video.write(annotated_frame)
        cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()


def main():
    video_filename = "VID20240503235720.mp4"

    obj_detector = object_detector()
    obj_detector(video_filename)


if __name__ == "__main__":
    main()

# %%
