from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import os
import uvicorn
import threading
import asyncio

# Check if running in a hosted environment
ON_RENDER = os.getenv("RENDER") is not None

app = FastAPI()

# Enable CORS for frontend access (update with actual frontend URL if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (make sure 'best.pt' is present)
app.state.model = YOLO("best.pt")

# Use a video file instead of a webcam (Change to 0 for webcam)
VIDEO_PATH = 0  # Set to 0 for webcam
app.state.cap = cv2.VideoCapture(VIDEO_PATH)

# Global variables
detected_animals = []  # Stores detected animal details

def read_frames():
    """Continuously reads frames in a background thread to prevent timeouts."""
    global app
    while True:
        success, _ = app.state.cap.read()
        if not success:
            app.state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if ended

# Start background frame reader in a separate thread
threading.Thread(target=read_frames, daemon=True).start()

@app.get("/")
def read_root():
    return {"message": "Animal Detection API Running!"}

@app.get("/video")
def video_feed():
    """Stream video frames with bounding boxes drawn."""
    def generate_frames():
        while True:
            success, frame = app.state.cap.read()
            if not success:
                print("⚠️ Frame not received! Retrying...")
                continue

            # Run YOLO detection
            results = app.state.model.predict(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = float(box.conf[0])  # Confidence score
                    class_id = int(box.cls[0])  # Class index
                    class_name = app.state.model.names[class_id]  # Get class name

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detect")
def get_detections():
    """Returns JSON response of detected animals."""
    return JSONResponse(content={"detections": detected_animals})

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket for real-time video streaming (alternative approach)."""
    await websocket.accept()
    
    while True:
        success, frame = app.state.cap.read()
        if not success:
            app.state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run YOLO detection
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

# Start FastAPI server
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # Default port 10000
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
