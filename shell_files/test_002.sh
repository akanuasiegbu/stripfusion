python  test_video.py \
  --weights ./runs/train/train_with_deform_conv_cosine_simirity_fix_from_scratch/weights/last.pt \
  --data  ./data/multispectral_temporal/kaist_video_test.yaml \
  --name train_with_deform_conv_cosine_simirity_fix_from_scratch2 \
  --lframe 3 \
  --temporal_stride 3 \
  --gframe 0 \
  --regex_search .set...V... \
  --dataset_used kaist \
  --img-size 640 \
  --save-json \
  --use_tadaconv \
  --detection_head lastframe \
  --task test \
  --exist-ok \
  --multiple_outputs \
  --conf-thres 0.001


  python  test_video.py \
  --weights ./runs/train/train_with_deform_conv_cosine_simirity_fix_from_scratch/weights/last.pt \
  --data  ./data/multispectral_temporal/kaist_video_test.yaml \
  --name train_with_deform_conv_cosine_simirity_fix_from_scratch2 \
  --lframe 3 \
  --temporal_stride 3 \
  --gframe 0 \
  --regex_search .set...V... \
  --dataset_used kaist \
  --img-size 640 \
  --save-json \
  --use_tadaconv \
  --detection_head lastframe \
  --task test \
  --exist-ok \
  --multiple_outputs \
  --conf-thres 0.2
