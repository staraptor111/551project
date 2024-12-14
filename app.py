from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
from newton4 import EngineClassifier
import torch.nn as nn
import torchvision.models as models
# Flask app 
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("classproj","static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining transformations 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading the original model (EngineClassifier)
MODEL_PATH = "models/best_model.pth"
model = EngineClassifier()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


MOBILENET_MODEL_PATH = "models/best_mobilenet_model.pth"


mobilenet_model = models.mobilenet_v2(pretrained=False)
mobilenet_model.classifier[-1] = nn.Linear(mobilenet_model.classifier[-1].in_features, 1)


state_dict = torch.load(MOBILENET_MODEL_PATH, map_location=device)
mobilenet_model.load_state_dict(state_dict)
mobilenet_model.to(device)
mobilenet_model.eval()

#  classification threshold calculation
ENGINE_THRESHOLD = 0.08
MOBILENET_ENGINE_THRESHOLD = 0.5 

def normalize_confidence(probability, threshold=0.08):
    if probability >= threshold:
        # for "Engine" confidence
        confidence = 50 + 50 * (probability - threshold) / (1 - threshold)
    else:
        # "Non-Engine" confidence
        confidence = 50 - 50 * (threshold - probability) / threshold
    return round(confidence, 2)  # Return confidence rounded to 2 decimal places

def classify_image(image_path, transform, model, device, threshold):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = model(image).squeeze()
            probability = torch.sigmoid(output).item()
            confidence = normalize_confidence(probability, threshold=threshold)
            classification = "Engine" if probability >= threshold else "Non-Engine"
            return probability, confidence, classification
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Classification
            probability_main, confidence_main, classification_main = classify_image(
                file_path, transform, model, device, threshold=ENGINE_THRESHOLD
            )
            
            probability_mobilenet, confidence_mobilenet, classification_mobilenet = classify_image(
                file_path, transform, mobilenet_model, device, threshold=MOBILENET_ENGINE_THRESHOLD
            )
            
            if probability_main is None or probability_mobilenet is None:
                return render_template("index.html", error="Error processing the uploaded image.")
            
            # Rendering a result page showing results from both models
            return render_template(
                "result.html", 
                filename=file.filename,
                classification_main=classification_main, 
                probability_main=probability_main, 
                confidence_main=confidence_main,
                classification_mobilenet=classification_mobilenet,
                probability_mobilenet=probability_mobilenet,
                confidence_mobilenet=confidence_mobilenet
            )
    
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))



if __name__ == "__main__":
    app.run(debug=True)
