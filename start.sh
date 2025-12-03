INPUT=$1
BASE_DIR="/home/truecut/kousei/Depth-Anything-3/gallery/depth"
i=1
EXPORT_DIR="${BASE_DIR}/$i"

# 如果目录存在就递增
while [ -d "$EXPORT_DIR" ]; do
    i=$((i+1))
    EXPORT_DIR="${BASE_DIR}/$i"
done

echo "输出目录: $EXPORT_DIR"
/home/truecut/anaconda3/envs/mm/bin/da3 auto "$INPUT" \
    --export-format exr\
    --process-res 1920\
    --export-dir "$EXPORT_DIR" \
    --use-backend