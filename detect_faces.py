from pathlib import Path
import cv2
import MtcnnDetector

def draw_bb(image, bb):
    (x, y, w, h) = bb
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_pts(image, pts):
    (lex, ley, rex, rey) = pts
    cv2.circle(image, ((int)(lex), (int)(ley)), 4, (0, 0, 255), -1)
    cv2.circle(image, ((int)(rex), (int)(rey)), 4, (0, 255, 0), -1)



FDet = MtcnnDetector.MtcnnDetector()

p = Path('Database_real_and_fake_face_160x160')

for imageFile in list(p.glob('**/*.jpg')):

    # Loading the image to be tested in grayscale
    image = cv2.imread(str(imageFile))

    # Face detection
    faces = FDet.detect_faces(image)
    if faces:
        face_bb, eyes_pts, shape = FDet.detect_largest_face(faces)

        # Let's draw a rectangle around the detected faces
        draw_bb(image, face_bb)
        draw_pts(image, eyes_pts)
        # Convert image to RGB and show image
        cv2.imshow('img', image)
        cv2.waitKey(500)
    
        
#normalizatorHS = faceutils.Normalization()
