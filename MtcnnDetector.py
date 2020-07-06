# /****************************************************************************************
#   Modesto Castrillón Santana
#   January 2008-2018
# ****************************************************************************************/
from mtcnn.mtcnn import MTCNN
import numpy as np

class MtcnnDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image):
        return self.detector.detect_faces(image)

    def detect_largest_face(self, faces):
        index = self.get_largest_bb(faces)
        if len(faces) < 1:
            return None
        # largest face
        face_info = faces[index]
        [x, y, w, h] = face_info['box']
        le = face_info['keypoints']['left_eye']
        re = face_info['keypoints']['right_eye']

        return [x, y, w, h], [le[0], le[1], re[0], re[1]], [
                face_info['keypoints']['left_eye'],
                face_info['keypoints']['right_eye'],
                face_info['keypoints']['nose'],
                face_info['keypoints']['mouth_left'],
                face_info['keypoints']['mouth_right']]

    def get_largest_bb(self, objects):
        if len(objects) < 1:
            return -1
        elif len(objects) == 1:
            return 0
        else:
            areas = [ (det['box'][2]*det['box'][3]) for det in objects ]
            return np.argmax(areas)

