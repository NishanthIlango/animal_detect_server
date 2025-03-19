from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
import asyncio
import os
import uvicorn

# Check if running in Render (prevents webcam error)
ON_RENDER = os.getenv("RENDER") is not None

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# OpenCV Video Capture (Only if running locally)
cap = None
if not ON_RENDER:
    cap = cv2.VideoCapture(0)  # Access camera only when running locally

detected_animal = ""
frame_lock = asyncio.Lock()

async def process_frame():
    """Runs continuously to update detected_animal."""
    global detected_animal
    while True:
        if cap is not None:
            success, frame = cap.read()
            if not success:
                await asyncio.sleep(0.1)  # Prevents busy-wait loop
                continue

            results = model.predict(frame)
            for result in results:
                for box in result.boxes:
                    detected_animal = "Detected"
        await asyncio.sleep(0.1)  # Prevents infinite loop CPU overload

@app.get("/")
def read_root():
    return {"message": "Animal Detection API Running!"}

@app.get("/video")
def video_feed():
    def generate_frames():
        while True:
            if cap is not None:
                success, frame = cap.read()
                if not success:
                    continue
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Start FastAPI server with correct port binding
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # Default to port 10000
    uvicorn.run(app, host="0.0.0.0", port=PORT)
