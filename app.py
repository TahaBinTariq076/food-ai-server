from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

app = FastAPI()

# ================== FOOD DATABASE ==================
FOOD_DB = {
    "chicken_wings": {"calories":430,"protein":30,"carbs":5,"fat":32,"ingredients":["chicken","spices","oil"]},
    "donuts": {"calories":300,"protein":4,"carbs":35,"fat":15,"ingredients":["flour","sugar","oil"]},
    "french_fries": {"calories":365,"protein":4,"carbs":63,"fat":17,"ingredients":["potatoes","oil","salt"]},
    "hummus": {"calories":180,"protein":8,"carbs":14,"fat":10,"ingredients":["chickpeas","tahini","olive oil"]},
    "ice_cream": {"calories":210,"protein":4,"carbs":24,"fat":11,"ingredients":["milk","cream","sugar"]},
    "red_velvet_cake": {"calories":450,"protein":5,"carbs":65,"fat":20,"ingredients":["flour","sugar","butter"]},
    "samosa": {"calories":260,"protein":6,"carbs":30,"fat":14,"ingredients":["potato","peas","flour","oil"]},
    "spring_rolls": {"calories":180,"protein":5,"carbs":20,"fat":8,"ingredients":["vegetables","wrapper","oil"]},
    "steak": {"calories":679,"protein":62,"carbs":0,"fat":48,"ingredients":["beef","salt","pepper"]},
    "waffles": {"calories":320,"protein":8,"carbs":40,"fat":15,"ingredients":["flour","milk","eggs"]}
}

# ================== LOAD MODELS ==================
print("🔥 Loading models...")

yolo = YOLO("best.pt")

CLASS_NAMES = list(FOOD_DB.keys())

model = models.convnext_small(weights=None)
model.classifier[2] = nn.Linear(
    model.classifier[2].in_features, len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load("ConvNeXt-Small_RC-Saliency.pt", map_location="cpu")
)

model.eval()

print("✅ Models loaded!")

# ================== TRANSFORM ==================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])

# ================== ROOT ==================
@app.get("/")
def home():
    return {"message": "Server is running ✅"}

# ================== PREDICT ==================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return JSONResponse({"error": "Invalid image"})

        results = yolo(img, conf=0.5)[0]

        if not results.boxes:
            return JSONResponse({"result": "No food detected"})

        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        tensor = transform(crop).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)

        conf, idx = torch.max(probs, 1)

        food_name = CLASS_NAMES[idx.item()]
        info = FOOD_DB.get(food_name, {})

        return JSONResponse({
            "food": food_name,
            "confidence": float(conf.item()),
            "calories": info.get("calories", 0),
            "protein": info.get("protein", 0),
            "carbs": info.get("carbs", 0),
            "fat": info.get("fat", 0),
            "ingredients": info.get("ingredients", [])
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})