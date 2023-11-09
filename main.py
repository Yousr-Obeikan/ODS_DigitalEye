from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import datetime
import httpx
from PIL import Image
import base64
from io import BytesIO


app = FastAPI()
report_url = "https://hodhod-dev-bk-core-testing.azurewebsites.net/apiSmartMonitoring/Voilation/voilationFromVendor"
headers = {"HodHodApiKey": "pgH7QzFHJx4w46fI~5Uzi4RvtTwlEXp"}

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
print("Model Loaded")
print("Classes", model.names)

@app.websocket("/ws")
async def process_frames(websocket: WebSocket):
    await websocket.accept()
    while True:

        # Receive frame from client as bytes
        data = await websocket.receive_bytes()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        # Process frame with YOLOv8 model
        results = model(frame)[0]
        
        # Assuming results contains the desired results
        xyxy = results.boxes.xyxy.tolist()
        conf = results.boxes.conf.tolist()
        cls = results.boxes.cls.tolist()
        
        # convert cls list of floats to list of ints
        cls = list(map(int, cls))
        
        # conver cls to class names using results.names
        cls = [results.names[i] for i in cls]
        
        # report send here
        for i in cls:
            if i == "NO-Hardhat" or i == "no_hairnet" or i == "smoke":
                resulted_image = Image.fromarray(results.plot()[..., ::-1])
                buffered = BytesIO()
                resulted_image.save(buffered, format="JPEG")
                data = buffered.getvalue()
                image_base64 = "!!" + base64.b64encode(data).decode('utf-8') + "!"
                time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                data = {
                    "cameraId": 4,
                    # send the image as base64 string
                    "picture": image_base64,
                    # get the current time in this format "2023-11-08T14:37:37.853Z"
                    "timeDate": time,
                    "voilationName": i,
                }

                response = httpx.post(report_url,headers=headers,json=data)
                print(response.json())
                break


        # Send back the results as json
        await websocket.send_json({"xyxy": xyxy, "conf": conf, "cls": cls})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
