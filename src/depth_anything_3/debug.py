import sys
from depth_anything_3.cli import app
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(app())

# auto /home/truecut/kousei/data/optical_flow_datasets/Unreal/train/clean/Matinee_03/24fps/image \
#     --export-format exr\
#     --process-res 1920\
#     --export-dir "/home/truecut/kousei/Depth-Anything-3/gallery/depth/test" \
#     --use-backend