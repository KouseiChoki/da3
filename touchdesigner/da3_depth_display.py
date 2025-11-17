"""
Depth Map Display Script TOP

Decodes base64 JPEG depth maps received from the server and displays them.

Usage:
1. Create a Script TOP named 'depth_display'
2. Paste this code
3. The script will automatically fetch depth data from parent storage

Note: Uses only built-in TouchDesigner modules (no PIL required)
Strategy: Save base64 to temp file, load with Movie File In TOP
"""

import base64
import tempfile
import os


def onSetupParameters(scriptOp):
    """Setup custom parameters."""
    page = scriptOp.appendCustomPage('Depth Display')

    page.appendToggle('Colorize', label='Colorize Depth', default=True)
    page.appendMenu('Colormap', label='Color Map', menuNames=[
        'viridis',
        'plasma',
        'inferno',
        'magma',
        'turbo',
        'gray'
    ], menuLabels=[
        'Viridis (Blue-Green-Yellow)',
        'Plasma (Purple-Pink-Yellow)',
        'Inferno (Black-Red-Yellow)',
        'Magma (Black-Purple-White)',
        'Turbo (Rainbow)',
        'Grayscale'
    ])
    page.appendToggle('Invert', label='Invert Depth', default=False)
    page.appendFloat('Brightness', label='Brightness', default=1.0, min=0.0, max=3.0)
    page.appendFloat('Contrast', label='Contrast', default=1.0, min=0.0, max=3.0)


def onCook(scriptOp):
    """
    Called when the Script TOP needs to update.
    Fetches depth data from storage and renders it.
    """
    # Get the latest depth data from parent storage
    depth_data = parent().fetch('depth_data', None)

    if depth_data is None:
        # No data yet, create black image
        scriptOp.clear()
        return

    try:
        # Decode base64 JPEG
        depth_bytes = base64.b64decode(depth_data)

        # Write to temp file
        temp_path = parent().fetch('depth_temp_path', None)
        if temp_path is None:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix='da3_depth_')
            os.close(temp_fd)  # Close file descriptor
            parent().store('depth_temp_path', temp_path)

        with open(temp_path, 'wb') as f:
            f.write(depth_bytes)

        # Load the image using a Movie File In TOP
        movie_top = scriptOp.parent().op('depth_loader_movie')
        if movie_top is None:
            # Create Movie File In TOP if it doesn't exist
            movie_top = scriptOp.parent().create(moviefileinTOP, 'depth_loader_movie')
            movie_top.par.file = temp_path

        # Update file path and reload
        movie_top.par.file = temp_path
        movie_top.par.reload.pulse()

        # Get pixel data from Movie File In TOP
        if movie_top.width > 0 and movie_top.height > 0:
            # Use TOP to CHOP to get pixel data, then process
            # For now, just copy the TOP output
            scriptOp.copy(movie_top)

            # Apply post-processing if needed
            if scriptOp.par.Invert.eval():
                # Use Composite TOP or Level TOP for processing
                # For simplicity, just copy for now
                pass

    except Exception as e:
        print(f"ERROR: Error in Script TOP: {e}")
        import traceback
        traceback.print_exc()
        scriptOp.clear()


def onDestroy(scriptOp):
    """Clean up temp file on destroy."""
    temp_path = parent().fetch('depth_temp_path', None)
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except:
            pass
