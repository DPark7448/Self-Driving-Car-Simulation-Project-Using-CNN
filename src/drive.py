import argparse
import base64
from collections import deque
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
from PIL import Image
import socketio
from flask import Flask
import tensorflow as tf

from preprocessing import preprocess


class SteeringSmoother:
    #Low-pass + rate limiter to follow intent quickly without snap.

    def __init__(self, alpha: float = 0.55, max_delta: float = 0.30):
        self.alpha = alpha
        self.max_delta = max_delta
        self.value = 0.0

    def update(self, target: float) -> float:
        blended = self.alpha * target + (1 - self.alpha) * self.value
        delta = np.clip(blended - self.value, -self.max_delta, self.max_delta)
        self.value += delta
        return self.value


def adaptive_throttle(base: float, steering: float, floor: float = 0.08) -> float:
    #Reduce throttle as steering grows to give time to rotate.
    scale = 1.0 - min(abs(steering), 1.0) * 1.0
    if abs(steering) > 0.25:
        scale *= 0.82
    if abs(steering) > 0.50:
        scale *= 0.68
    return max(base * scale, floor)


sio = socketio.Server(cors_allowed_origins="*")
app = Flask(__name__)
MAX_PRED_STEER = 1.0


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def make_server(model, base_throttle: float):
    sio.smoother = SteeringSmoother()
    sio.pred_buffer = deque(maxlen=3)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if not data or "image" not in data:
            print(f"sid={sid} missing telemetry payload")
            return

        img_str = data.get("image")
        steering_angle = 0.0
        throttle = base_throttle
        raw_angle = 0.0

        try:
            image = Image.open(BytesIO(base64.b64decode(img_str)))
            image_array = np.asarray(image)
            proc = preprocess(image_array)

            raw_pred = float(model.predict(proc[None, ...], verbose=0)[0][0])
            raw_angle = float(np.clip(raw_pred, -MAX_PRED_STEER, MAX_PRED_STEER))

            # Light median smoothing only.
            sio.pred_buffer.append(raw_angle)
            raw_angle = float(np.median(sio.pred_buffer))

            gain = 1.25 if raw_angle >= 0 else 1.35
            steering_cmd = raw_angle * gain
            steering_cmd = float(np.clip(steering_cmd, -MAX_PRED_STEER, MAX_PRED_STEER))

            steering_angle = sio.smoother.update(steering_cmd)
            steering_angle = float(np.clip(steering_angle, -MAX_PRED_STEER, MAX_PRED_STEER))

            throttle = adaptive_throttle(base_throttle, steering_angle)

        except Exception as exc:
            print(f"sid={sid} failed to process telemetry: {exc}")
            return

        print(f"sid={sid} raw={raw_angle:.4f} steer={steering_angle:.4f} throttle={throttle:.2f}")
        sio.emit(
            "steer",
            data={"steering_angle": str(steering_angle), "throttle": str(throttle)},
            room=sid,
        )

    @sio.on("connect")
    def connect(sid, environ):
        print("Simulator connected:", sid)
        sio.emit("steer", data={"steering_angle": "0.0", "throttle": str(base_throttle)}, room=sid)

    return telemetry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.keras")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--throttle", type=float, default=0.10, help="Constant throttle during drive")
    args = parser.parse_args()

    model = load_model(args.model_path)
    make_server(model, args.throttle)
    app_wrapped = socketio.Middleware(sio, app)
    print(f"Serving on http://{args.host}:{args.port}, waiting for simulator...")
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app_wrapped)


if __name__ == "__main__":
    main()
