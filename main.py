from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# Load the YOLOv8 model
model = YOLO('model/best.pt')

@app.websocket("/ws")
async def process_frames(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive frame from client
        data = await websocket.receive_text()
        frame_bytes = base64.b64decode(data.split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Process frame with YOLOv8 model
        results_list = model(frame)
        
        # Assuming results_list[0] contains the desired results
        xyxy = results_list[0].boxes.xyxy
        conf = results_list[0].boxes.conf
        cls = results_list[0].boxes.cls
        
        # Send back the results
        await websocket.send_text(f"{xyxy.tolist()},{conf.tolist()},{cls.tolist()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
