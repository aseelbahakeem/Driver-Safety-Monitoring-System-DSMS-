
"""
Real-Time Drowsiness Detection Webcam Testing
Requirements:
pip install torch torchvision opencv-python pillow
"""

import cv2
import torch
from torchvision import transforms
from PIL import Image
import time

# Configuration
MODEL_PATH = 'best_model.pth'  # Download this from Colab
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
import timm
model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)
print(f"Model loaded on {DEVICE}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Face detector (optional - for cropping face)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_frame(frame):
    """Predict drowsiness from frame"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    
    return pred.item(), confidence.item()

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit()

print("Webcam opened successfully!")
print("Press 'q' to quit")

fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect face 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Use full frame for prediction
    pred, conf = predict_frame(frame)
    
    # Labels
    label = "DROWSY" if pred == 0 else "ALERT"
    color = (0, 0, 255) if pred == 0 else (0, 255, 0)  # Red for drowsy, Green for alert
    
    # Draw rectangle around face if detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Display prediction
    text = f"{label} ({conf*100:.1f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3)
    
    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
        print(f"Prediction: {label} | Confidence: {conf*100:.1f}% | FPS: {fps:.1f}")
    
    # Show frame
    cv2.imshow('Drowsiness Detection - Press Q to quit', frame)
    
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
