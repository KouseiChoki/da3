# export MODEL_DIR=/home/truecut/kousei/Depth-Anything-3/checkpoints
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=/home/truecut/kousei/Depth-Anything-3/gallery
export CUDA_VISIBLE_DEVICES="0"
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend  --gallery-dir ${GALLERY_DIR} # Cache model to gpu
