"""
Frame Sender for DA3 Streaming (chopexecDAT) - Numpy Version

Sends frames as raw numpy arrays instead of JPEG for better performance.

Usage:
1. Create a chopexecDAT
2. Paste this code
3. Set the parameters below to match your setup
4. Connect to the CHOP you want to trigger frame sends (e.g., a Timer CHOP)
"""


def onOffToOn(channel, sampleIndex, val, prev):
    """
    Called when channel value goes from 0 to non-zero.
    Use this with a Timer CHOP set to pulse mode.
    """
    send_frame()


def onValueChange(channel, sampleIndex, val, prev):
    """
    Alternative: send on every value change.
    Use this with a constant timer.
    """
    # Uncomment to send every frame instead of on pulse:
    # send_frame()
    pass


def send_frame():
    """
    Capture frame from TOP and send to WebSocket as raw numpy array.
    """
    # Configuration
    control_script = op('da3_stream_control')
    websocket_dat = op('websocket1')  # Adjust name if needed
    source_top = op('in_video')  # Video input TOP

    # Get frame skip setting
    frame_skip = control_script.par.Frameskip.eval()

    # Check if we should skip this frame
    frame_counter = parent().fetch('frame_counter', 0)
    parent().store('frame_counter', frame_counter + 1)

    if frame_counter % frame_skip != 0:
        return  # Skip this frame

    # Check if WebSocket is connected
    if not websocket_dat.par.active.eval():
        debug("WARNING: WebSocket not connected, skipping frame")
        return

    # Check if we should send as numpy or JPEG
    use_numpy = control_script.par.Usenumpy.eval()

    try:
        if use_numpy:
            # Send as raw numpy array (faster, no compression artifacts)
            import numpy as np
            import json

            # Get numpy array from TOP
            # Method 1: Use numpyArray() if available
            try:
                frame_array = source_top.numpyArray(delayed=False)
                # TouchDesigner returns (height, width, channels) in float32 [0, 1]
                # Convert to uint8 [0, 255]
                frame_uint8 = (frame_array * 255).astype(np.uint8)

                h, w, c = frame_uint8.shape

                # Send JSON metadata first
                metadata = {
                    "format": "numpy",
                    "shape": [h, w, c],
                    "dtype": "uint8"
                }
                websocket_dat.sendText(json.dumps(metadata))

                # Then send binary data
                websocket_dat.sendBinary(frame_uint8.tobytes())

                debug(f"Sent numpy frame: {h}x{w}x{c}")

            except AttributeError:
                # numpyArray() not available, fall back to JPEG
                debug("WARNING: numpyArray() not available, falling back to JPEG")
                send_frame_jpeg(source_top, websocket_dat, control_script)

        else:
            # Send as JPEG (smaller, but lossy)
            send_frame_jpeg(source_top, websocket_dat, control_script)

        # Update stats
        sent_count = parent().fetch('frames_sent', 0)
        parent().store('frames_sent', sent_count + 1)

    except Exception as e:
        debug(f"ERROR: Error sending frame: {e}")
        import traceback
        traceback.print_exc()


def send_frame_jpeg(source_top, websocket_dat, control_script):
    """
    Send frame as JPEG (fallback method).
    """
    # Get JPEG quality from control script (or use default 85)
    try:
        quality = control_script.par.Quality.eval()
        if quality == 100:
            quality = 85  # JPEG doesn't support 100 well
        quality_float = quality / 100.0  # Convert to 0.0-1.0
    except:
        quality_float = 0.85

    # Convert TOP to JPEG bytes
    jpeg_bytes = source_top.saveByteArray('.jpg', quality=quality_float)

    # Send via WebSocket as binary
    websocket_dat.sendBinary(jpeg_bytes)

    debug(f"Sent JPEG frame: {len(jpeg_bytes)} bytes")


def debug(msg):
    """Print debug message."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Frame Sender: {msg}")
