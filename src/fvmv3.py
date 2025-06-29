import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity

class FingerVeinMatcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._initialize_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _initialize_model(self):
        """Initialize ResNet model for feature extraction"""
        model = resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for grayscale
        model.fc = nn.Identity()  # Remove classification layer
        model.eval()
        return model.to(self.device)
    
    def preprocess(self, image):
        """Enhance vein patterns and prepare image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing pipeline
        image = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.resize(image, (224, 224))
    
    def extract_features(self, image):
        """Extract deep features from finger vein image"""
        # Preprocess and transform
        processed = self.preprocess(image)
        tensor = self.transform(processed).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()
    
    def match_images(self, img1, img2):
        """
        Compare two finger vein images and return similarity score
        :param img1: First image (file path or numpy array)
        :param img2: Second image (file path or numpy array)
        :return: Similarity score between 0-1 (1 = identical)
        """
        # Handle input types
        image1 = cv2.imread(img1) if isinstance(img1, str) else img1
        image2 = cv2.imread(img2) if isinstance(img2, str) else img2
        
        # Feature extraction
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        
        # Calculate cosine similarity
        return cosine_similarity([feat1], [feat2])[0][0]

# Example usage
if __name__ == "__main__":
    matcher = FingerVeinMatcher()
    
    # Compare two images
    score = matcher.match_images("./input/genuine/1393-PLUS-FV3-Laser_PALMAR_060_01_07_01.png", "./input/genuine/1390-PLUS-FV3-Laser_PALMAR_060_01_04_03.png")
    print(f"Matching score: {score:.4f}")
    
    # Expected output range: 0.0 to 1.0
    # Typical interpretation:
    #   < 0.6 = Different fingers
    #   0.6-0.8 = Potential match (adjust threshold)
    #   > 0.8 = High confidence match