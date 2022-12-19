import numpy as np
import cv2


def source_landmark_points():
    src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32)
    
    return src
    


def align_faces(img, bbox=None, landmark=None, **kwargs):
    M = None
    # Do alignment using landmark points
    if landmark is not None:
        src = source_landmark_points()
        src[:, 0] += 25.0
        src[:, 1] += 15.0
        dst = landmark.astype(np.float32)
        # Created the transformation matrix
        M = cv2.estimateAffine2D(dst,src)[0]
        # Apply 
        warped = cv2.warpAffine(img,M,(150,150), borderValue = 0.0)
        # Return The tranformation Matrix too
        return warped, M
    
    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        x1,y1,x2,y2,_ = bbox
        ret = img[y1:y2,x1:x2]
        ret = cv2.resize(ret, (112,112))
        return ret, None

def face_distance(vec1,vec2):
    return np.dot(vec1, vec2.T)
