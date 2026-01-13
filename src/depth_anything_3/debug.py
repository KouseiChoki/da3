import sys,os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from depth_anything_3.cli import app
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(app())
#  auto /Users/qhong/Downloads/rgb.mp4 --device mps --export-dir /Users/qhong/Desktop/0112 --export-format exr
# auto /Users/qhong/Desktop/1104/archar/image --device mps --export-dir /Users/qhong/Desktop/0113 --export-format exr --process-res 1920 --maxframe 10
# auto /home/truecut/kousei/data/optical_flow_datasets/Unreal/train/clean/Matinee_03/24fps/image \
#     --export-format exr\
#     --process-res 1920\
#     --export-dir "/home/truecut/kousei/Depth-Anything-3/gallery/depth/test" \
#     --use-backend