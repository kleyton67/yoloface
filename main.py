from face_detector import YoloDetector
import numpy as np
from PIL import Image

model = YoloDetector(target_size=720,gpu=0,min_face=90)
orgimg = np.array(Image.open('photo_2022-12-13_08-38-58.jpg'))
bboxes,points = model.predict(orgimg)