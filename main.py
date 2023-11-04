from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

user_selected_model = int(input("""Select a model:
1 - Hardhat
2 - Smoke
3 - Hairnet
"""))
MODELS = {
    1: "model/best_hardhat.pt",
    2: "model/best_smoke.pt",
    3: "model/best_hairnet.pt"
}
# Load the YOLOv8 model
model = YOLO(MODELS[user_selected_model])

@app.websocket("/ws")
async def process_frames(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive frame from client
        data = await websocket.receive_text()
        frame_bytes = base64.b64decode(data.split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        # other option Receive frame from client as bytes
        # data = await websocket.receive_bytes()
        # frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        # Process frame with YOLOv8 model
        results_list = model(frame)[0]
        
        # Assuming results_list contains the desired results
        xyxy = results_list.boxes.xyxy.tolist()
        conf = results_list.boxes.conf.tolist()
        cls = results_list.boxes.cls.tolist()
        
        # convert cls list of floats to list of ints
        cls = list(map(int, cls))
        
        # conver cls to class names using results_list.names
        cls = [results_list.names[i] for i in cls]
        
        # Send back the results as json
        await websocket.send_json({"xyxy": xyxy, "conf": conf, "cls": cls})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
