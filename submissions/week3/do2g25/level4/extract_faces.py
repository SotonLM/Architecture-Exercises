# Importing the libraries
import cv2
import os


# Define a class to extract the faces
class ExtractFaces:
    def __init__(self):
        # Initialise the face detector
        self.classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face(self, img):
        # Convert the image to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        x, y, w, h = faces

        return x, y, w, h

    def extract_face_with_margin(self, image_path, save_path, margin_percent=20):
        # Read the image
        img = cv2.imread(image_path)

        # Get the face
        x, y, w, h = self.detect_face(img)

        # Calculate margins
        margin_w = int(w * (margin_percent / 100))
        margin_h = int(h * (margin_percent / 100))

        # Add margins with boundary checking
        new_x = max(0, x - margin_w)
        new_y = max(0, y - margin_h)
        new_w = min(img.shape[1] - new_x, w + 2 * margin_w)
        new_h = min(img.shape[0] - new_y, h + 2 * margin_h)

        # Extract face with margins
        face = img[new_y:new_y + new_h, new_x:new_x + new_w]

        cv2.imwrite(save_path, face)


face_extractor = ExtractFaces()

for file in os.listdir("face_data/positive_samples"):
    extracted_face_path = f"face_data/just_face_positive_samples/{file}"
    face_extractor.extract_face_with_margin(f"face_data/positive_samples/{file}", extracted_face_path)




