from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
from ultralytics import YOLO
import os
import uvicorn
import asyncio

# Check if running in Render (prevents webcam error)
ON_RENDER = os.getenv("RENDER") is not None

app = FastAPI()

# Load YOLO model once (instead of reloading on every request)
app.state.model = YOLO("best.pt")

# Global variables
app.state.detected_animals = []  # List to store detected animal details
app.state.cap = None if ON_RENDER else cv2.VideoCapture(0)  # Open camera globally

async def process_frame():
    """Continuously updates detected_animals asynchronously."""
    cap = app.state.cap
    model = app.state.model

    try:
        while True:
            if cap and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    await asyncio.sleep(0.1)
                    continue

                results = model.predict(frame)
                detected_animals = []

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        confidence = float(box.conf[0])  # Confidence score
                        class_id = int(box.cls[0])  # Class index
                        class_name = model.names[class_id]  # Get class name

                        detected_animals.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        })

                app.state.detected_animals = detected_animals  # Update global detection list

            await asyncio.sleep(0.1)  # Prevent high CPU usage
    finally:
        if cap:
            cap.release()

@app.on_event("startup")
async def startup_event():
    """Starts the background task for processing frames."""
    asyncio.create_task(process_frame())

@app.on_event("shutdown")
async def shutdown_event():
    """Releases the camera resource on shutdown."""
    if app.state.cap:
        app.state.cap.release()

@app.get("/")
def read_root():
    return {"message": "Animal Detection API Running!"}

@app.get("/video")
def video_feed():
    """Stream video frames with bounding boxes drawn."""
    def generate_frames():
        cap = app.state.cap
        model = app.state.model

        try:
            while True:
                if cap and cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        continue

                    results = model.predict(frame)
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]

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
    return JSONResponse(content={"detections": app.state.detected_animals})

# Start FastAPI server
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # Default to port 10000
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
