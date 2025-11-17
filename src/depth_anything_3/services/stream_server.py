# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Streaming Server for Real-time Depth Estimation

Provides HTTP and WebSocket endpoints for real-time depth estimation.
Optimized for integration with TouchDesigner, vvvv, Processing, etc.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import json
from typing import Optional
import asyncio
from contextlib import asynccontextmanager
import time

from ..api import DepthAnything3
from .streaming import StreamingDepthEstimator, StreamConfig
from ..utils.sliding_window import WindowConfig


# Global state
stream_estimator: Optional[StreamingDepthEstimator] = None
model: Optional[DepthAnything3] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model, stream_estimator
    # Startup
    print("🚀 Starting Depth Anything 3 Streaming Server...")
    yield
    # Shutdown
    print("👋 Shutting down streaming server...")


app = FastAPI(
    title="Depth Anything 3 Streaming API",
    description="Real-time depth estimation for creative coding",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_model(model_dir: str, device: str, process_res: int = 504):
    """Initialize the model."""
    global model, stream_estimator
    print(f"📦 Loading model from {model_dir} on {device}...")
    print(f"⚙️  Process resolution: {process_res}")
    model = DepthAnything3.from_pretrained(model_dir).to(device)
    stream_estimator = StreamingDepthEstimator(model, device=device, process_res=process_res)
    print(f"✅ Model loaded and ready for streaming")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Depth Anything 3 Streaming API",
        "version": "1.0.0",
        "status": "running" if model is not None else "not initialized",
        "endpoints": {
            "http_frame": "POST /process/frame - Process single frame (HTTP)",
            "websocket": "WS /stream - Real-time streaming (WebSocket)",
            "stats": "GET /stats - Get streaming statistics",
            "health": "GET /health - Health check",
        },
        "touchdesigner": "Use WebSocket endpoint for real-time streaming",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "not initialized",
        "model_loaded": model is not None,
        "device": stream_estimator.device if stream_estimator else None,
    }


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify server is running."""
    return {
        "message": "Server is running!",
        "timestamp": time.time(),
        "websocket_endpoint": "/stream",
    }


@app.get("/stats")
async def get_stats():
    """Get streaming statistics."""
    if stream_estimator is None:
        return {"error": "Streaming not initialized"}
    return stream_estimator.get_stats()


@app.post("/process/frame")
async def process_frame_http(file: UploadFile = File(...), format: str = "numpy"):
    """
    Process a single frame via HTTP.

    Args:
        file: Image file (JPEG, PNG, etc.)
        format: Output format - "numpy" (base64 encoded), "png", or "jpeg"

    Returns:
        Depth map in requested format
    """
    if stream_estimator is None:
        return JSONResponse({"error": "Model not initialized"}, status_code=503)

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    depth = stream_estimator.process_frame(frame)

    if depth is None:
        return {"status": "buffering", "message": "Frame buffered, waiting for more frames"}

    # Convert depth to requested format
    if format == "png" or format == "jpeg":
        # Normalize to 0-255
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

        # Encode as image
        if format == "png":
            _, buffer = cv2.imencode('.png', depth_norm)
        else:
            _, buffer = cv2.imencode('.jpg', depth_norm, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return Response(content=buffer.tobytes(), media_type=f"image/{format}")
    else:  # numpy format
        # Base64 encode numpy array
        depth_bytes = depth.tobytes()
        depth_b64 = base64.b64encode(depth_bytes).decode('utf-8')

        return {
            "status": "success",
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "data": depth_b64,
        }


@app.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    window_size: int = Query(default=None, description="Window size for sliding window processing"),
    overlap: int = Query(default=None, description="Overlap between windows"),
    buffer_size: int = Query(default=None, description="Buffer size before processing"),
    max_fps: float = Query(default=None, description="Maximum FPS (throttles processing)"),
    quality: int = Query(default=85, description="JPEG quality for output (0-100)"),
):
    """
    WebSocket endpoint for real-time streaming.

    Query Parameters:
        - window_size: Number of frames per processing window (default: auto based on device)
        - overlap: Number of overlapping frames between windows (default: window_size // 3)
        - buffer_size: Minimum frames before processing (default: window_size // 2)
        - max_fps: Maximum processing FPS (default: unlimited)
        - quality: JPEG quality for output depth maps (0-100, default: 85)

    Protocol:
        Client -> Server: Binary image data (JPEG/PNG)
        Server -> Client: JSON with base64 encoded depth map

    Example (TouchDesigner):
        1. Add WebSocket DAT
        2. Set Address to: ws://localhost:8080/stream?window_size=8&max_fps=15
        3. Send JPEG frames using webSocketDAT.sendBinary(jpeg_bytes)
        4. Receive depth maps via callbacks DAT
    """
    print("=" * 60)
    print("🔌 NEW WebSocket connection incoming...")

    try:
        await websocket.accept()
        print("✅ WebSocket connection ACCEPTED")
    except Exception as e:
        print(f"❌ Failed to accept WebSocket connection: {e}")
        return

    # Parse configuration from query params
    device = stream_estimator.device if stream_estimator else "mps"

    # Create per-connection stream estimator with custom config
    if window_size is not None or overlap is not None or buffer_size is not None:
        print(f"📝 Creating custom config from query params:")
        print(f"   window_size={window_size}, overlap={overlap}, buffer_size={buffer_size}")

        # StreamingDepthEstimator uses buffer_size as window_size
        config = StreamConfig.for_device(device)

        if window_size is not None:
            config.buffer_size = window_size
        if overlap is not None:
            config.overlap = overlap
        if buffer_size is not None:
            config.buffer_size = buffer_size

        # Create custom estimator for this connection
        per_conn_estimator = StreamingDepthEstimator(
            model,
            device=device,
            config=config,
        )
    else:
        # Use global estimator
        print(f"📝 Using default config")
        per_conn_estimator = stream_estimator

    print(f"⚙️  Active Config:")
    print(f"   - Buffer size: {per_conn_estimator.config.buffer_size}")
    print(f"   - Overlap: {per_conn_estimator.config.overlap}")
    print(f"   - Output latency: {per_conn_estimator.config.output_latency_frames}")
    print(f"   - Max FPS: {max_fps or 'unlimited'}")
    print(f"   - JPEG quality: {quality}")
    print("=" * 60)

    if per_conn_estimator is None:
        print("❌ ERROR: Estimator is None!")
        await websocket.send_json({"error": "Model not initialized"})
        await websocket.close()
        return

    # FPS throttling
    min_frame_time = 1.0 / max_fps if max_fps else 0
    last_frame_time = 0
    frame_count = 0

    try:
        print("👂 Waiting for frames from client...")

        while True:
            # Receive image data
            print(f"\n🔄 Waiting for frame #{frame_count + 1}...")

            try:
                # Try to receive any message (text or binary)
                message = await websocket.receive()
                msg_type = message.get('type')

                if msg_type == 'websocket.receive':
                    if 'bytes' in message:
                        data = message['bytes']
                        print(f"Received BINARY: {len(data)} bytes")
                    elif 'text' in message:
                        text = message['text']
                        # Check if it's numpy metadata (JSON)
                        try:
                            metadata = json.loads(text)
                            if metadata.get('format') == 'numpy':
                                # Expect binary data next
                                bin_msg = await websocket.receive()
                                if 'bytes' in bin_msg:
                                    data = bin_msg['bytes']
                                    print(f"Received numpy: {len(data)} bytes")
                                else:
                                    print("Expected binary after numpy metadata")
                                    continue
                            else:
                                await websocket.send_json({
                                    "status": "error",
                                    "message": "Send binary image data or numpy format"
                                })
                                continue
                        except json.JSONDecodeError:
                            await websocket.send_json({
                                "status": "error",
                                "message": "Invalid JSON metadata"
                            })
                            continue
                    else:
                        print(f"Unknown message format: {message}")
                        continue
                else:
                    print(f"Unexpected message type: {msg_type}")
                    continue

            except Exception as e:
                print(f"Error receiving data: {e}")
                import traceback
                traceback.print_exc()
                break

            frame_count += 1

            # FPS throttling
            if max_fps:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < min_frame_time:
                    # Skip frame
                    print(f"Frame #{frame_count} THROTTLED (max_fps={max_fps})")
                    await websocket.send_json({
                        "status": "throttled",
                        "message": f"Frame skipped (max_fps={max_fps})",
                    })
                    continue
                last_frame_time = current_time

            # Decode image
            print(f"Decoding frame #{frame_count}...")

            # Try numpy format first (more efficient)
            if 'metadata' in locals() and metadata.get('format') == 'numpy':
                try:
                    h, w, c = metadata['shape']
                    dtype = metadata.get('dtype', 'uint8')
                    frame = np.frombuffer(data, dtype=dtype).reshape((h, w, c))
                    print(f"Frame #{frame_count} decoded from numpy: {frame.shape}")
                except Exception as e:
                    print(f"Failed to decode numpy: {e}")
                    await websocket.send_json({"error": "Failed to decode numpy"})
                    continue
            else:
                # Fall back to JPEG decoding
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"Failed to decode frame #{frame_count}")
                    await websocket.send_json({"error": "Failed to decode image"})
                    continue

                print(f"Frame #{frame_count} decoded from JPEG: {frame.shape}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            print(f"🧠 Processing frame #{frame_count}...")
            depth = per_conn_estimator.process_frame(frame)

            if depth is None:
                # Still buffering
                stats = per_conn_estimator.get_stats()
                buffer_fill = stats.get('frames_in_buffer', 0)
                print(f"⏳ BUFFERING... ({buffer_fill}/{per_conn_estimator.config.buffer_size} frames)")
                await websocket.send_json({
                    "status": "buffering",
                    "stats": stats,
                })
                continue

            print(f"✅ Depth computed: {depth.shape}")

            # Option 1: Send as JPEG (smaller, uint8, requires decoding)
            if quality < 100:
                # Normalize to 0-255 for JPEG
                depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                _, buffer = cv2.imencode('.jpg', depth_norm, [cv2.IMWRITE_JPEG_QUALITY, quality])
                depth_b64 = base64.b64encode(buffer).decode('utf-8')
                data_format = "jpeg_base64"
                depth_min = float(depth.min())
                depth_max = float(depth.max())
            else:
                # Option 2: Send raw float32 (larger, no normalization, no decoding)
                depth_float32 = depth.astype(np.float32)
                depth_b64 = base64.b64encode(depth_float32.tobytes()).decode('utf-8')
                data_format = "raw_float32"
                depth_min = float(depth.min())
                depth_max = float(depth.max())

            print(f"Sending depth map ({len(depth_b64)} chars base64, format={data_format})")

            # Send response
            await websocket.send_json({
                "status": "success",
                "shape": list(depth.shape),
                "dtype": "float32" if data_format == "raw_float32" else "uint8",
                "format": data_format,
                "data": depth_b64,
                "min": depth_min,
                "max": depth_max,
                "stats": per_conn_estimator.get_stats(),
            })

            print(f"✅ Frame #{frame_count} complete!")

    except WebSocketDisconnect:
        print(f"\n👋 WebSocket client disconnected (processed {frame_count} frames)")
    except Exception as e:
        print(f"\n❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass


def start_stream_server(
    model_dir: str = "depth-anything/DA3-BASE",
    device: str = "mps",
    host: str = "0.0.0.0",
    port: int = 8080,
    process_res: int = 504,
):
    """
    Start the streaming server.

    Args:
        model_dir: Model directory or HuggingFace repo
        device: Device to use (mps, cuda, cpu)
        host: Host to bind to
        port: Port to bind to
        process_res: Processing resolution (will be rounded to nearest multiple of 14)
    """
    import uvicorn

    # Initialize model
    init_model(model_dir, device, process_res)

    print(f"")
    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"📡 WebSocket endpoint: ws://{host}:{port}/stream")
    print(f"🔗 HTTP endpoint: http://{host}:{port}/process/frame")
    print(f"")
    print(f"💡 TouchDesigner Integration:")
    print(f"   1. Use Web Client DAT to connect to ws://{host}:{port}/stream")
    print(f"   2. Send JPEG frames via WebSocket")
    print(f"   3. Receive depth maps in real-time")
    print(f"")

    # Start server
    uvicorn.run(app, host=host, port=port, log_level="info")
