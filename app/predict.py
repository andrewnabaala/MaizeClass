import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load the trained model
def load_model(model_path):
    # Define the model architecture
    model = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust for 4 output classes

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict the class of the image
def predict(image_path, model):
    classes = ['Healthy', 'Grey_leaf_spot', 'Common_rust', 'Blight']
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]