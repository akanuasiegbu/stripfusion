

# CUDA_VISIBLE_DEVICES='6' \
# python kaist_to_json.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1.yaml \
#     --batch-size 8 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search .set...V... \
#     --temporal_mosaic \
#     --image_set train

# CUDA_VISIBLE_DEVICES='6' \
# python kaist_to_json.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1.yaml \
#     --batch-size 8 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search .set...V... \
#     --temporal_mosaic \
#     --image_set val


CUDA_VISIBLE_DEVICES='7' \
python kaist_to_json.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1_midframe.yaml \
    --batch-size 8 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
    --lframe 1 \
    --temporal_stride 1 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search .set...V... \
    --temporal_mosaic \
    --image_set train \
    --midframe

CUDA_VISIBLE_DEVICES='7' \
python kaist_to_json.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1_midframe.yaml \
    --batch-size 8 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
    --lframe 1 \
    --temporal_stride 1 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search .set...V... \
    --temporal_mosaic \
    --image_set val \
    --midframe
