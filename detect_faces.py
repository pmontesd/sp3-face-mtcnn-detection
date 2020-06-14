from pathlib import Path
import cv2
import MtcnnDetector
import FaceNormalizationUtils as faceutils

def draw_bb(image, bb):
    (x, y, w, h) = bb
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_pts(image, pts):
    (lex, ley, rex, rey) = pts
    cv2.circle(image, ((int)(lex), (int)(ley)), 4, (0, 0, 255), -1)
    cv2.circle(image, ((int)(rex), (int)(rey)), 4, (0, 255, 0), -1)

def normalize_color_image(image, pts):
    (lex, ley, rex, rey) = pts
    # Fr HS normalization
    B, G, R = cv2.split(image)
    normalizator.normalize_gray_img(B, lex, ley, rex, rey, faceutils.Kind_wraping.FACE)
    Bnorm = normalizator.normf_image
    normalizator.normalize_gray_img(G, lex, ley, rex, rey, faceutils.Kind_wraping.FACE)
    Gnorm = normalizator.normf_image
    normalizator.normalize_gray_img(R, lex, ley, rex, rey, faceutils.Kind_wraping.FACE)
    Rnorm = normalizator.normf_image
    return cv2.merge((Bnorm, Gnorm, Rnorm))


FDet = MtcnnDetector.MtcnnDetector()
normalizator = faceutils.Normalization()

p = Path('Database_real_and_fake_face_160x160')

for imageFile in list(p.glob('**/*.jpg')):

    # Loading the image to be tested in grayscale
    image = cv2.imread(str(imageFile))

    # Face detection
    faces = FDet.detect_faces(image)
    if faces:
        face_bb, eyes_pts, shape = FDet.detect_largest_face(faces)

        # Normalize and show color channels
        normalized_image = normalize_color_image(image, eyes_pts)
        # Convert image to RGB and show image
        cv2.imshow("Normalized", normalized_image)
        cv2.waitKey(100)
    else:
        print(F"No face detected in {imageFile}", file=sys.stderr) 
