"""
Depth Anything 3 Streaming Control for TouchDesigner (Raw Numpy Version)

Uses raw numpy arrays instead of JPEG encoding for better performance.

Usage:
1. Create a Text DAT named 'da3_stream_control'
2. Paste this code
3. Create a WebSocket DAT and set its Callbacks DAT to this script
4. Access parameters via op('da3_stream_control').par.ParameterName

Parameters will appear in the parameter dialog automatically.
"""


def onSetupParameters(scriptOp):
    """
    Define custom parameters for the streaming control.
    Called when the operator is created or reset.
    """
    page = scriptOp.appendCustomPage('DA3 Stream')

    # Connection Settings
    page.appendStr('Host', label='Host', default='localhost')
    page.appendStr('Port', label='Port', default='8080')

    # Model Configuration
    page.appendMenu('Model', label='Model', menuNames=[
        'DA3-SMALL',
        'DA3-BASE',
        'DA3-LARGE',
        'DA3-GIANT',
        'DA3NESTED-GIANT-LARGE'
    ], menuLabels=[
        'DA3-SMALL (80M - Fastest)',
        'DA3-BASE (120M - Balanced)',
        'DA3-LARGE (350M - Quality)',
        'DA3-GIANT (1.15B - Slow)',
        'DA3NESTED-GIANT-LARGE (1.4B - Metric+GS)'
    ])

    page.appendMenu('Device', label='Device', menuNames=[
        'mps',
        'cuda',
        'cpu'
    ], menuLabels=[
        'MPS (Apple Silicon)',
        'CUDA (NVIDIA)',
        'CPU (Slow)'
    ])

    # Stream Configuration
    page.appendInt('Windowsize', label='Window Size', default=8, min=1, max=32)
    page.appendInt('Overlap', label='Overlap', default=2, min=0, max=16)
    page.appendInt('Processres', label='Process Resolution', default=504, min=252, max=1008)
    page.appendFloat('Maxfps', label='Max FPS', default=15.0, min=1.0, max=60.0)
    page.appendInt('Quality', label='Quality (100=raw float)', default=100, min=50, max=100)

    # Input Settings
    page.appendInt('Frameskip', label='Frame Skip (N)', default=1, min=1, max=10)
    page.appendToggle('Usenumpy', label='Send as Numpy', default=True)

    # Backend Management
    page.appendStr('Backendpath', label='Backend Path',
                   default='/Users/flo/work/code/Depth-Anything-3')
    page.appendStr('Modeldir', label='Model Directory',
                   default='depth-anything/DA3-BASE')
    page.appendInt('Backendport', label='Backend Port', default=8080)
    page.appendToggle('Autorestart', label='Auto Restart Backend', default=True)

    # Control
    page.appendPulse('Reconnect', label='Reconnect')
    page.appendPulse('Restartbackend', label='Restart Backend')
    page.appendPulse('Stopbackend', label='Stop Backend')

    # Status (read-only)
    page.appendStr('Status', label='Status', default='Disconnected', readOnly=True)
    page.appendStr('Backendstatus', label='Backend Status', default='Unknown', readOnly=True)


def onPulse(par):
    """
    Called when a pulse parameter is triggered.
    """
    scriptOp = par.owner

    if par.name == 'Reconnect':
        reconnect(scriptOp)
    elif par.name == 'Restartbackend':
        restart_backend(scriptOp)
    elif par.name == 'Stopbackend':
        stop_backend(scriptOp)


def onValueChange(par):
    """
    Called when any parameter value changes.
    Auto-reconnect if configured and model/device changes.
    """
    scriptOp = par.owner

    # Auto-restart backend if model or device changes
    if scriptOp.par.Autorestart.eval():
        if par.name in ['Model', 'Device', 'Modeldir']:
            debug(f"WARNING: {par.name} changed, restarting backend...")
            restart_backend(scriptOp)
            # Wait a bit then reconnect
            run("op('da3_stream_control').reconnect_delayed()", delayFrames=120)


def reconnect_delayed():
    """Delayed reconnect helper (called after backend restart)."""
    reconnect(op('da3_stream_control'))


def build_websocket_url(scriptOp):
    """
    Build WebSocket URL from parameters.

    Returns:
        str: WebSocket URL with query parameters
    """
    host = scriptOp.par.Host.eval()
    port = scriptOp.par.Port.eval()

    # Build query parameters
    params = []
    params.append(f"window_size={scriptOp.par.Windowsize.eval()}")
    params.append(f"overlap={scriptOp.par.Overlap.eval()}")
    params.append(f"max_fps={scriptOp.par.Maxfps.eval()}")
    params.append(f"quality={scriptOp.par.Quality.eval()}")

    query_string = '&'.join(params)
    url = f"ws://{host}:{port}/stream?{query_string}"

    debug(f"WebSocket URL: {url}")
    return url


def reconnect(scriptOp):
    """
    Reconnect the WebSocket with current parameters.
    """
    ws = find_websocket_dat(scriptOp)
    if not ws:
        debug("ERROR: No WebSocket DAT found!")
        scriptOp.par.Status = "Error: No WebSocket"
        return

    # Disconnect if connected
    if ws.par.active.eval():
        debug("Disconnecting existing WebSocket...")
        ws.par.active = False
        # Wait a frame before reconnecting
        run("op('da3_stream_control').do_connect()", delayFrames=30)
    else:
        do_connect()


def do_connect():
    """Actually perform the connection (called after disconnect delay)."""
    scriptOp = op('da3_stream_control')
    ws = find_websocket_dat(scriptOp)

    if not ws:
        return

    # Build URL
    url = build_websocket_url(scriptOp)

    # Update WebSocket settings
    ws.par.address = url
    ws.par.autoReconnect = True
    ws.par.active = True

    scriptOp.par.Status = "Connecting..."
    debug("Reconnecting with new settings...")


def find_websocket_dat(scriptOp):
    """
    Find the WebSocket DAT that uses this script as callbacks.

    Returns:
        OP or None: WebSocket DAT operator
    """
    # Look for a WebSocket DAT in the same parent
    parent = scriptOp.parent()
    for child in parent.children:
        if child.type == 'websocketDAT':
            # Check if it uses us as callbacks
            if child.par.callbacks and child.par.callbacks.eval() == scriptOp:
                return child

    # Fallback: look for common names
    candidates = ['websocket1', 'ws_depth', 'da3_websocket']
    for name in candidates:
        ws = parent.op(name)
        if ws and ws.type == 'websocketDAT':
            return ws

    return None


def restart_backend(scriptOp):
    """
    Restart the streaming backend server.
    """
    import subprocess
    import os

    scriptOp.par.Backendstatus = "Restarting..."

    # Stop existing backend
    stop_backend(scriptOp)

    # Wait a bit
    import time
    time.sleep(1.0)

    # Build command
    backend_path = scriptOp.par.Backendpath.eval()
    model = scriptOp.par.Model.eval()
    device = scriptOp.par.Device.eval()
    port = scriptOp.par.Backendport.eval()
    process_res = scriptOp.par.Processres.eval()

    # Map model menu to actual model directory
    model_map = {
        'DA3-SMALL': 'depth-anything/DA3-SMALL',
        'DA3-BASE': 'depth-anything/DA3-BASE',
        'DA3-LARGE': 'depth-anything/DA3-LARGE',
        'DA3-GIANT': 'depth-anything/DA3-GIANT',
        'DA3NESTED-GIANT-LARGE': 'depth-anything/DA3NESTED-GIANT-LARGE'
    }
    model_dir = model_map.get(model, scriptOp.par.Modeldir.eval())

    # Activate venv and run da3 stream
    activate_script = os.path.join(backend_path, '.venv', 'bin', 'activate')
    cmd = f'source "{activate_script}" && cd "{backend_path}" && da3 stream --model-dir {model_dir} --device {device} --port {port} --process-res {process_res} --host 0.0.0.0'

    debug(f"Starting backend: {cmd}")

    try:
        # Store process handle in storage
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )

        parent().store('da3_backend_process', proc)
        scriptOp.par.Backendstatus = f"Running (PID: {proc.pid})"
        debug(f"Backend started (PID: {proc.pid})")

    except Exception as e:
        scriptOp.par.Backendstatus = f"Error: {str(e)}"
        debug(f"ERROR: Failed to start backend: {e}")


def stop_backend(scriptOp):
    """
    Stop the streaming backend server.
    """
    import signal
    import os

    proc = parent().fetch('da3_backend_process', None)

    if proc and proc.poll() is None:  # Process is running
        try:
            # Kill process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            debug(f"Stopped backend (PID: {proc.pid})")
            scriptOp.par.Backendstatus = "Stopped"
        except Exception as e:
            debug(f"ERROR: Error stopping backend: {e}")
            scriptOp.par.Backendstatus = f"Error: {str(e)}"
    else:
        scriptOp.par.Backendstatus = "Not running"
        debug("WARNING: Backend not running")


def check_backend_health(scriptOp):
    """
    Check if backend is responsive.

    Returns:
        bool: True if healthy
    """
    import urllib.request
    import json

    host = scriptOp.par.Host.eval()
    port = scriptOp.par.Port.eval()

    try:
        url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read())
            if data.get('status') == 'healthy':
                scriptOp.par.Backendstatus = f"Healthy ({data.get('model', 'unknown')})"
                return True
    except Exception as e:
        debug(f"WARNING: Backend health check failed: {e}")
        scriptOp.par.Backendstatus = "Unreachable"

    return False


# WebSocket Callbacks

def onOpen(dat):
    """Called when WebSocket connection opens."""
    scriptOp = op('da3_stream_control')
    scriptOp.par.Status = "Connected"
    debug("WebSocket connected!")


def onClose(dat):
    """Called when WebSocket connection closes."""
    scriptOp = op('da3_stream_control')
    scriptOp.par.Status = "Disconnected"
    debug("WebSocket disconnected")


def onReceiveText(dat, message):
    """
    Called when text/JSON message is received from server.
    """
    import json
    import base64
    import numpy as np

    try:
        data = json.loads(message)

        if data.get('status') == 'success':
            # Get format
            data_format = data.get('format', 'jpeg_base64')
            depth_b64 = data['data']
            shape = tuple(data['shape'])
            depth_min = data.get('min', 0.0)
            depth_max = data.get('max', 1.0)

            # Decode based on format
            if data_format == 'raw_float32':
                # Raw float32 numpy array
                depth_bytes = base64.b64decode(depth_b64)
                depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(shape)
                debug(f"Received raw float32 depth: {shape}, range={depth_min:.3f}-{depth_max:.3f}")
            else:
                # JPEG uint8 - need to denormalize
                depth_bytes = base64.b64decode(depth_b64)
                depth_uint8 = np.frombuffer(depth_bytes, dtype=np.uint8).reshape(shape)
                # Denormalize from 0-255 back to original range
                depth = (depth_uint8.astype(np.float32) / 255.0) * (depth_max - depth_min) + depth_min
                debug(f"Received JPEG depth: {shape}, denormalized to range={depth.min():.3f}-{depth.max():.3f}")

            # Store depth array in parent storage
            parent().store('depth_array', depth)
            parent().store('depth_min', depth_min)
            parent().store('depth_max', depth_max)

            # Trigger depth display Script TOP to cook
            depth_display = op('depth_display')
            if depth_display:
                depth_display.cook(force=True)

            # Update status with stats
            stats = data.get('stats', {})
            fps = stats.get('fps', 0)
            op('da3_stream_control').par.Status = f"Streaming @ {fps:.1f} FPS"

        elif data.get('status') == 'buffering':
            stats = data.get('stats', {})
            buffer_fill = stats.get('frames_in_buffer', 0)
            op('da3_stream_control').par.Status = f"Buffering ({buffer_fill} frames)"

        elif data.get('status') == 'error':
            error_msg = data.get('message', 'Unknown error')
            op('da3_stream_control').par.Status = f"Error: {error_msg}"
            debug(f"ERROR: Server error: {error_msg}")

    except Exception as e:
        debug(f"ERROR: Error processing message: {e}")
        import traceback
        traceback.print_exc()


def onReceiveBinary(dat, bytes):
    """Called when binary message is received (unexpected)."""
    debug(f"Received unexpected binary: {len(bytes)} bytes")


# Utility

def debug(msg):
    """Print debug message with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] DA3 Stream: {msg}")


# Initialize default values
def onCook(scriptOp):
    """Called every frame - use for periodic tasks."""
    # Check backend health every 2 seconds
    if scriptOp.time.frame % 60 == 0:  # Assuming 30fps = every 2 sec
        check_backend_health(scriptOp)
