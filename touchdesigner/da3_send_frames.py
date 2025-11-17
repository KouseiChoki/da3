"""
Frame Sender for DA3 Streaming (chopexecDAT)

This script sends frames from a TOP (video input, movie file, camera, etc.)
to the DA3 streaming server via WebSocket.

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
    Capture frame from TOP and send to WebSocket.
    """
    # Configuration
    control_script = op('da3_stream_control')
    websocket_dat = op.TDResources.op('websocket1')  # Adjust name if needed
    source_top = op('moviefilein1')  # Change to your video source TOP

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

    # Get JPEG quality from control script
    quality = control_script.par.Quality.eval() / 100.0  # Convert to 0.0-1.0

    try:
        # Convert TOP to JPEG bytes
        jpeg_bytes = source_top.saveByteArray('.jpg', quality=quality)

        # Send via WebSocket as binary
        websocket_dat.sendBinary(jpeg_bytes)

        # Update stats
        sent_count = parent().fetch('frames_sent', 0)
        parent().store('frames_sent', sent_count + 1)

    except Exception as e:
        debug(f"ERROR: Error sending frame: {e}")
        import traceback
        traceback.print_exc()


def debug(msg):
    """Print debug message."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Frame Sender: {msg}")
