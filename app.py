import asyncio
import base64
import json
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sam3_engine import SAM3Engine

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/web", StaticFiles(directory="web"), name="web")

engine = SAM3Engine()


@app.get("/")
async def get():
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    queue = asyncio.Queue(maxsize=1)
    stop_event = threading.Event()

    # Task to consume queue and send to websocket
    async def sender():
        try:
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await websocket.send_text(json.dumps(data))
                    queue.task_done()
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"Sender error: {e}")

    sender_task = asyncio.create_task(sender())
    video_thread = None

    def video_worker(source, loop):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Failed to open video source: {source}")
            return

        frame_idx = 0
        rgb_times = []
        engine.init_session(mode=engine.mode)

        while not stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_combined = engine.process_frame(frame_rgb, frame_idx)

            end_time = time.time()
            rgb_times.append(end_time - start_time)
            if len(rgb_times) > 30:
                rgb_times.pop(0)

            rgb_fps = 1.0 / (sum(rgb_times) / len(rgb_times)) if rgb_times else 0
            seg_fps = engine.get_seg_fps()

            result_bgr = cv2.cvtColor(result_combined, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            stats = {
                "type": "frame",
                "image": jpg_as_text,
                "stats": {
                    "rgb_fps": round(rgb_fps, 1),
                    "seg_fps": round(seg_fps, 1),
                    "mode": engine.mode,
                    "prompt_summary": {
                        "text": engine.text_prompt,
                        "points": len(engine.points),
                        "boxes": len(engine.boxes),
                    },
                },
            }

            if not queue.full():
                loop.call_soon_threadsafe(queue.put_nowait, stats)

            frame_idx += 1

        cap.release()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "start":
                source = payload.get("source", "0")
                if source == "webcam" or source == "0":
                    source = 0

                if video_thread:
                    stop_event.set()
                    video_thread.join()
                    stop_event.clear()

                loop = asyncio.get_event_loop()
                video_thread = threading.Thread(
                    target=video_worker, args=(source, loop), daemon=True
                )
                video_thread.start()

            elif msg_type == "set_text":
                engine.set_text_prompt(payload.get("text", ""))
            elif msg_type == "add_point":
                engine.add_point(payload.get("x"), payload.get("y"), payload.get("label"))
            elif msg_type == "add_box":
                engine.add_box(
                    payload.get("x1"),
                    payload.get("y1"),
                    payload.get("x2"),
                    payload.get("y2"),
                    payload.get("label"),
                )
            elif msg_type == "clear_prompts":
                engine.clear_prompts()
            elif msg_type == "set_thresholds":
                engine.score_threshold = payload.get("score_threshold", 0.5)
                engine.mask_threshold = payload.get("mask_threshold", 0.5)

    except WebSocketDisconnect:
        stop_event.set()
        if video_thread:
            video_thread.join()
        sender_task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
