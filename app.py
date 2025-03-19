from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
import asyncio

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
detected_animal = ""
frame_lock = asyncio.Lock()


async def process_frame():
    """Runs continuously to update detected_animal."""
    global detected_animal
    while True:
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(0.1)  # Prevents busy-wait loop
            continue

        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                animal = model.names[cls]
                detected_label = f"{animal} walking" if animal == "tiger" else animal

                async with frame_lock:
                    detected_animal = detected_label  # Thread-safe update

        await asyncio.sleep(0.1)  # Avoid excessive CPU usage


async def generate_frames():
    """Stream video frames with detection."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detection")
async def get_detection():
    async with frame_lock:
        return {"animal": detected_animal}


@app.on_event("startup")
async def startup_event():
    """Starts background frame processing when the app starts."""
    asyncio.create_task(process_frame())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on shutdown."""
    cap.release()
