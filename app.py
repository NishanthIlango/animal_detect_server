from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import os
import uvicorn
import asyncio

# Check if running in Render (prevents webcam error)
ON_RENDER = os.getenv("RENDER") is not None

app = FastAPI()

# Enable CORS (fixes issues with hosted access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once (instead of reloading on every request)
app.state.model = YOLO("best.pt")

# Use a video file instead of a webcam (fix for hosted servers)
VIDEO_PATH = "sample_video.mp4"  # Upload this file to your server
app.state.cap = cv2.VideoCapture(VIDEO_PATH)

# Global variables
detected_animals = []  # List to store detected animal details

async def process_frame():
    """Continuously updates detected_animals asynchronously."""
    global detected_animals
    cap = None if ON_RENDER else cv2.VideoCapture(VIDEO_PATH)  # Use video file instead of webcam

    try:
        while True:
            if cap and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if ended
                    continue

                results = app.state.model.predict(frame)

                detected_animals = []  # Reset detected animals
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        confidence = float(box.conf[0])  # Confidence score
                        class_id = int(box.cls[0])  # Class index
                        class_name = app.state.model.names[class_id]  # Get class name

                        detected_animals.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        })

            await asyncio.sleep(0.1)  # Prevents high CPU usage
    finally:
        if cap:
            cap.release()

@app.get("/")
def read_root():
    return {"message": "Animal Detection API Running!"}

@app.get("/video")
def video_feed():
    """Stream video frames with bounding boxes drawn."""
    def generate_frames():
        cap = None if ON_RENDER else cv2.VideoCapture(VIDEO_PATH)
        
        try:
            while True:
                if cap and cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if ended
                        continue

                    results = app.state.model.predict(frame)
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = app.state.model.names[class_id]

                            # Draw bounding box & label on frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name} {confidence:.2f}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    _, buffer = cv2.imencode(".jpg", frame)
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        finally:
            if cap:
                cap.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detect")
def get_detections():
    """Returns JSON response of detected animals."""
    return JSONResponse(content={"detections": detected_animals})

# WebSocket for alternative video streaming (optional)
@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket for real-time video streaming (alternative approach)."""
    await websocket.accept()
    cap = cv2.VideoCapture(VIDEO_PATH)

    try:
        while True:
            if cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                results = app.state.model.predict(frame)

                # Draw bounding boxes
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = app.state.model.names[class_id]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                _, buffer = cv2.imencode(".jpg", frame)
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.1)
    finally:
        cap.release()
        await websocket.close()

# Start FastAPI server
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # Default to port 10000
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
