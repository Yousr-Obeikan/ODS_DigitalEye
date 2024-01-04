import base64
import datetime
import threading
from io import BytesIO

import cv2
from PIL import Image
from ultralytics import YOLO

report_url = "https://hodhod-dev-bk-core-testing.azurewebsites.net/apiSmartMonitoring/Voilation/voilationFromVendor"
headers = {"HodHodApiKey": "pgH7QzFHJx4w46fI~5Uzi4RvtTwlEXp"}

# user_selected_model = int(input("""Select a model:
# 1 - Hardhat
# 2 - Smoke
# 3 - Hairnet
# """))

# Initialize set for notified track IDs
notified_track_ids = set()

# User-selected model
user_selected_model = 1

MODELS = {
    1: "model/best_hardhat.pt",
    2: "model/best_smoke.pt",
    3: "model/best_hairnet.pt"
}

# Load the YOLOv8 model
model = YOLO(MODELS[user_selected_model])
print("Model Loaded")
print("Classes", model.names)

# Setup RTSP stream
camera_url = "rtsp://admin:pass123456@20.0.0.242:554/Streaming/channels/1"
cap = cv2.VideoCapture(0)

# Initialize a buffer for the latest frame
latest_frame = None


# Function to continuously update the latest frame
def update_frame():
    global latest_frame
    while True:
        success, frame = cap.read()
        if success:
            latest_frame = frame


# Start a separate thread to keep updating the latest frame
frame_update_thread = threading.Thread(target=update_frame, daemon=True)
frame_update_thread.start()

# Loop to process the latest frame
while True:
    if latest_frame is not None:
        # Process the latest frame with YOLOv8 model
        results = model.track(source=latest_frame, persist=True, tracker="custom_tracker.yaml")[0]

        # Visualize the results on the frame
        annotated_frame = results.plot()

        # Optional: Display the annotated frame with detection
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Get the boxes and track IDs
        try:
            boxes = results.boxes.xywh.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()
            # Check and notify for each track ID
            for box, track_id in zip(boxes, track_ids):
                if track_id not in notified_track_ids:
                    # Assuming results contains the desired results
                    cls = results.boxes.cls.tolist()
                    # convert cls list of floats to list of ints
                    cls = list(map(int, cls))
                    # convert cls to class names using results.names
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
                            data = {"cameraId": 4, "picture": image_base64, "timeDate": time, "voilationName": i, }

                            # response = httpx.post(report_url, headers=headers, json=data)
                            # print(response.json())

                            # Add the track ID to the notified set
                            notified_track_ids.add(track_id)
                            print("Notification sent")
                            break
                else:
                    print(f"Track {track_id} already exists")

        except Exception as e:
            print("No Detection")
            print(e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
