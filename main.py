# import dependencies
from IPython.display import display, Audio
from base64 import b64decode, b64encode
import cv2, PIL, io, os, html, time
import numpy as np
import torch
from gtts import gTTS
from ultralytics import YOLO
from googletrans import Translator
from supervision import Detections, LabelAnnotator, BoundingBoxAnnotator
import soundfile as sf
import librosa


class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoundingBoxAnnotator()
        self.label_annotator = LabelAnnotator()
        self.KNOWN_DISTANCE = 16 #INCHES
        self.MOBILE_WIDTH = 3.0 #INCHES
        res = self.model.predict('/Ref_images/person_mobile1.jpg', verbose=False)
        x1, y1, x2, y2 = res[0][1].boxes.xyxy.cpu().numpy()[0]
        # mobile width in ref frame (in pixels)
        self.mobile_width_in_rf = x2 - x1


    def load_model(self):
        model = YOLO("yolov8l.pt", task='detect')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model.predict(frame, verbose=False)
        return results

    def focal_length_finder (self, measured_distance, real_width, width_in_rf):
        focal_length = (width_in_rf * measured_distance) / real_width
        return focal_length

    # distance finder function
    def distance_finder(self, focal_length, real_object_width, width_in_frmae):
        distance = (real_object_width * focal_length) / width_in_frmae
        return int(round(distance,0))

    def mobile_width_in_frame(self, detections):
        for xyxy, mask, confidence, class_id, tracker_id, data in detections:
            if class_id==67:
                x1, y1, x2, y2 = xyxy
                width = x2 - x1
                return width

    def play_audio(self, text, ch):
        if ch==1:
            tts = gTTS(text=text, lang='en', tld="us")
            print(text)
        elif ch==2:
            translator = Translator()
            translated = translator.translate(text, src='en', dest='hi')
            tts = gTTS(text=translated.text, lang='hi', tld="us")
            print(translated.text)
        else:
            translator = Translator()
            translated = translator.translate(text, src='en', dest='mr')
            tts = gTTS(text=translated.text, lang='mr', tld="us")
            print(translated.text)

        tts.save('temp_audio.wav')
        audio, sr = librosa.load("temp_audio.wav")
        # Increase the speed by 1.5x
        faster_audio = librosa.effects.time_stretch(audio, rate=1.5)
        sf.write("temp_audio.wav", faster_audio, sr)
        display(Audio('temp_audio.wav', autoplay=True))
        time.sleep(2)

    def plot_bboxes(self, results, frame, ch):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections if conf > 0.8
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            if confidence > 0.75:
                xyxy = result.boxes.xyxy.cpu().numpy()
                xyxys.append(xyxy.reshape(-1, 4))
                confidences.append(confidence)
                class_ids.append(class_id)

        if xyxys:
            # Setup detections for visualization
            detections = Detections(
                xyxy=np.concatenate(xyxys),
                confidence=np.concatenate(confidences),
                class_id=np.concatenate(class_ids),
            )

            # Logic to check if mobile detected
            if any([True if class_id==67 else False for xyxy, mask, confidence, class_id, tracker_id, data in detections]):
                focal_mobile = self.focal_length_finder(self.KNOWN_DISTANCE, self.MOBILE_WIDTH, self.mobile_width_in_rf)
                width_in_frame = self.mobile_width_in_frame(detections)
                distance = self.distance_finder(focal_mobile, self.MOBILE_WIDTH, width_in_frame)
                # Format custom labels
                self.labels = [f"Mobile at Distance: {distance} inches" if class_id==67 else f'{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}' for xyxy, mask, confidence, class_id, tracker_id, data in detections]

                txt = f'Mobile is {distance} inches away'
                self.play_audio(txt, ch)

                # Annotate and display frame
                frame = self.box_annotator.annotate(scene=frame, detections=detections)
                frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
                return frame

            # Format custom labels
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, mask, confidence, class_id, tracker_id, data in detections]

            # Play audio sound
            if self.labels:
                if len(self.labels)==1:
                    for label in self.labels:
                        item = label.split()
                        if len(item)==3:
                            txt = f"There is a {item[0]} {item[1]} ahead"
                            self.play_audio(txt, ch)
                        elif len(item)==2:
                            txt = f"There is a {item[0]} ahead"
                            self.play_audio(txt, ch)
                else:
                    txt = f'There are {len(self.labels)} objects: '
                    for label in self.labels[:-1]:
                        item = label.split()
                        if len(item)==3:
                            txt += f"{item[0]} {item[1]}, "
                        elif len(item)==2:
                            txt += f"{item[0]}, "
                    # Adding last item
                    last_item = self.labels[-1].split()
                    if len(last_item)==3:
                        txt += f"and {last_item[0]} {last_item[1]}"
                    elif len(last_item)==2:
                        txt += f"and {last_item[0]}"
                    self.play_audio(txt, ch)

            # Annotate and display frame
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame

    def __call__(self):
        print("Choose Language...\n  1. For English\n  2. For Hindi\n  3. For Marathi\n")
        while True:
            try:
                ch = int(input('Enter Here (1, 2 or 3): '))
                if ch not in [1, 2, 3]:
                    print("Invalid input. Please enter 1, 2, or 3.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    detector = ObjectDetection(capture_index=0)
    detector()
    
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
#   5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
#   10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
#   15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
#   20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
#   25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
#   30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
#   35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
#   40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
#   45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
#   50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
#   55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
#   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
#   65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
#   70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
#   75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}