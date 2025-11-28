# Importing the libraries
import cv2
import torch
import torch.nn as nn
from transformers import ViTModel
import numpy as np


# Define the model architecture
class ViTForBinaryClassification(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTForBinaryClassification, self).__init__()

        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Get hidden size from ViT config
        hidden_size = self.vit.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)

        # Use the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0]

        # Pass through classifier
        logits = self.classifier(cls_output)

        return logits


class App:
    def __init__(self, authorise_match_amount=10, buffer_size=30, target_size=(224, 224)):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Pick image size for classification
        self.target_size = target_size

        # Load the face classifier
        self.classifier = ViTForBinaryClassification()
        self.classifier.load_state_dict(torch.load("face_recognition_vit_pytorch_V1.pth",
                                                   map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()  # Set to evaluation mode

        # Initialising the buffer tracking matches
        self.buffer = []

        # Choosing how many consecutive matches are needed to authorise
        self.authorise_match_amount = authorise_match_amount

        # Choosing a buffer size
        self.buffer_size = buffer_size

        # Track if the user is authenticated
        self.is_authenticated = False

    # Defining a method to track positive matches
    def track_positive_matches(self, is_positive):
        # Add the latest prediction
        self.buffer.append(is_positive)

        # Remove the oldest prediction when appropriate
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Check the buffer for consecutive matches
        consecutive_matches = 0
        for prediction in self.buffer:
            if prediction:
                # If the prediction is positive increment the counter
                consecutive_matches += 1
            elif prediction is None:
                # If the prediction is unsure then counter remains unchanged
                continue
            elif not prediction:
                # If the prediction is negative set the counter to zero
                consecutive_matches = 0

        # Check if there are enough consecutive matches to authorise
        if consecutive_matches >= self.authorise_match_amount:
            self.is_authenticated = True

    # Defining a method to detect a face
    def detect_face(self, frame):
        """Detect face in frame and return coordinates"""
        # Convert to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Returns the rectangle in which any faces are detected
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Return None if no faces are detected
        if len(faces) == 0:
            return None

        # Get the largest face in case multiple are detected
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        return x, y, w, h

    def preprocess_face(self, face):
        """Preprocess face image for PyTorch model"""
        # Normalize to [0, 1]
        face = face.astype(np.float32) / 255.0

        # Convert from HWC to CHW format (Height, Width, Channels -> Channels, Height, Width)
        face = np.transpose(face, (2, 0, 1))

        # Add batch dimension and convert to tensor
        face = torch.FloatTensor(face).unsqueeze(0)

        return face

    def gui(self):
        # Initialises video capture screen (0 specifies default camera of device)
        cap = cv2.VideoCapture(0)

        # If camera is not found
        if not cap.isOpened():
            # End the program
            print("Cannot open camera")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                # If ret is False then exit the loop
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get key press if there was any
            key = cv2.waitKey(1)

            # Check to see if the user wants to exit (pressed the 'q' key)
            if key == ord('q'):
                # Exit the loop
                break

            # Check to see if the user wants to reset authentication
            elif key == ord('r') and self.is_authenticated:
                self.is_authenticated = False
                self.buffer.clear()

            # Detect face
            if self.detect_face(frame) is None:
                continue
            else:
                x, y, w, h = self.detect_face(frame)

            # Find the margin around the face
            margin = int(0.2 * w)  # 20% margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)

            # Extract and resize face
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, self.target_size)

            # Preprocess face for PyTorch
            face_tensor = self.preprocess_face(face)
            face_tensor = face_tensor.to(self.device)

            # Classify the face
            with torch.no_grad():  # Disable gradient computation for inference
                prediction = self.classifier(face_tensor)
                prediction = prediction.cpu().numpy()  # Move back to CPU and convert to numpy

            # Check if there were enough positive matches in a row to authorise
            if prediction[0][0] <= 0.4 and not self.is_authenticated:
                # If the prediction was positive add True to the buffer
                self.track_positive_matches(True)
            elif 0.4 < prediction[0][0] < 0.6 and not self.is_authenticated:
                # If the prediction was unsure add None to the buffer
                self.track_positive_matches(None)
            elif prediction[0][0] >= 0.6 and not self.is_authenticated:
                # If the prediction was negative add False to the buffer
                self.track_positive_matches(False)

            # Display authorisation message if appropriate
            if self.is_authenticated:
                cv2.putText(frame, "Authorised", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(prediction)

            # Check if the prediction is strongly positive
            if prediction[0][0] <= 0.4:
                # If so put green rectangle over my face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # If the prediction is uncertain
            elif 0.4 < prediction[0][0] < 0.6:
                # Put yellow rectangle over face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Check if the prediction is strongly negative
            elif prediction[0][0] >= 0.6:
                # If so put red rectangle over face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Face Recognition App', frame)

        # When the loop is over, release the camera and close the app
        cap.release()
        cv2.destroyAllWindows()
        exit()


if __name__ == "__main__":
    app = App()
    app.gui()
