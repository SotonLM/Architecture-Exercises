# Importing the libraries
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
import random


# Defining a class to preprocess the LFW dataset
class LFWPreprocessor:
    # Constructor function
    def __init__(self,
                 lfw_dir="raw_LFW_dataset/lfw-deepfunneled",  # Directory containing LFW dataset
                 output_dir="face_data/negative_samples",
                 target_size=(224, 224),  # Standard size for face images
                 num_samples=1000):  # Number of faces to extract

        # Set attributes
        self.lfw_dir = Path(lfw_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.num_samples = num_samples

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # Defining a method to detect and align the face
    def detect_and_align_face(self, image):
        """Detect face in image and align it"""
        # Converting the image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Returns the rectangle in which any faces are detected
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Return None if no faces are detected
        if len(faces) == 0:
            return None

        # Get the largest face in case multiple are detected
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Add margin around face
        margin = int(0.2 * w)  # 20% margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Extract and resize face
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, self.target_size)

        return face

    # Defining a method to preprocess the dataset
    def process_dataset(self):
        """Process the LFW dataset"""
        # Return error if the specified directory where the dataset should be does not exist
        if not self.lfw_dir.exists():
            raise FileNotFoundError(f"LFW directory not found at {self.lfw_dir}")

        # Check if the output data already exists
        if self.output_dir.exists():
            # If so, remove it
            shutil.rmtree(self.output_dir)
        # Create output directory
        self.output_dir.mkdir(parents=True)

        # Initiate a list of all the paths to the individual images in LFW
        image_paths = []

        # Iterate over all the person directories in LFW
        for person_dir in self.lfw_dir.iterdir():
            # Check if the file being considered is a directory
            if person_dir.is_dir():
                # If so get all the files of that person and add them to the image_paths list
                image_paths.extend(list(person_dir.glob("*.jpg")))

        # Randomly sample images
        selected_paths = random.sample(image_paths, min(self.num_samples, len(image_paths)))

        # Counters to track progress
        successful = 0
        failed = 0

        # Output that images are being processed
        print(f"Processing {len(selected_paths)} images...")

        # Using tqdm to show a progress bar
        for idx, img_path in enumerate(tqdm(selected_paths)):
            try:
                # Read image
                image = cv2.imread(str(img_path))

                # Incrementing the failed counter if no image was read
                if image is None:
                    failed += 1
                    continue

                # Process face with detect and align method
                face = self.detect_and_align_face(image)

                # Incrementing the failed counter if no face was detected
                if face is None:
                    failed += 1
                    continue

                # Create numbered filename for detected faces
                output_path = self.output_dir / f"face_{idx:04d}.jpg"

                # Save the face and increment the successful counter
                cv2.imwrite(str(output_path), face)
                successful += 1

            # If there is an error, increment the failed counter and continue
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                failed += 1
                continue

        # Output information about the processing
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed to process: {failed} images")
        print(f"Processed images saved to: {self.output_dir}")

        # Returns True if at least one image was read
        return successful > 0


# Defining a function to process LFW
def preprocess_lfw():
    """Main function to preprocess LFW dataset"""
    # Creating an object of the LFW processor class
    processor = LFWPreprocessor()

    # Trying to process the dataset
    try:
        return processor.process_dataset()
    except Exception as e:
        print(f"Error preprocessing dataset: {str(e)}")
        return False


# Driver code
preprocess_lfw()
