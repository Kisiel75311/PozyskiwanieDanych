import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = torch.load('trained_model.pth', map_location=torch.device('cpu'))
model.eval()

# Load test image
# Load test image and convert to RGB to remove alpha channel if present
test_image = "test_image_3.jpg"
image = Image.open(test_image).convert('RGB')

# Rest of your code...


# Prepare transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform and add an extra dimension
input_image = transform(image).unsqueeze(0).to(device)

# Get result from model
outputs = model(input_image)

# Grad-CAM function to get the weights
def get_cam_weights(grads):
    return np.mean(grads, axis=(1, 2))

# Hook functions and storage for gradients and activations
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks on the last layer
last_layer = model.layer4[2].conv3
last_layer.register_forward_hook(forward_hook)
last_layer.register_backward_hook(backward_hook)

# Forward pass to capture activations
output = model(input_image)

# We index the output to get the score for the predicted class
predicted_class = output.argmax(dim=1)

# Backward pass to capture gradients
model.zero_grad()
predicted_class_score = output[0][predicted_class]
predicted_class_score.backward()

# Get captured gradients and activations
gradients_val = gradients[0].cpu().data.numpy().squeeze()
activations_val = activations[0].cpu().data.numpy().squeeze()

print("Gradients shape: ", gradients_val.shape)
print("Activations shape: ", activations_val.shape)


# Compute weights using global average pooling on gradients
cam_weights = get_cam_weights(gradients_val)

# Weighted combination of feature maps
cam_output = np.zeros((activations_val.shape[1], activations_val.shape[2]), dtype=np.float32)
for i, w in enumerate(cam_weights):
    cam_output += w * activations_val[i, :, :]

# Normalize CAM to be between 0 and 1
cam_output = np.maximum(cam_output, 0)
cam_output = cam_output - np.min(cam_output)
cam_output = cam_output / np.max(cam_output)

# Resize CAM to the size of the original image
cam_output = cv2.resize(cam_output, (image.width, image.height))

# Apply color map to the CAM
heatmap = cv2.applyColorMap(np.uint8(255 * cam_output), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

# Combine heatmap with original image
combined_image = heatmap + np.float32(image.convert('RGB')) / 255
combined_image = combined_image / np.max(combined_image)

# Save combined image
cv2.imwrite("cam_image.jpg", np.uint8(255 * combined_image))

print("Class Activation Map saved as cam_image.jpg")

# ... (previous code)

# Forward pass to capture activations
output = model(input_image)

# We index the output to get the score for the predicted class
predicted_class = output.argmax(dim=1).item()  # .item() to get the value as a Python scalar

# Translate the index to a class name for binary classification
predicted_class_name = 'Class 1' if predicted_class == 1 else 'Class 0'

# Print the class name
print(f"The model predicted: {predicted_class_name}")

# ... (the rest of your code)

