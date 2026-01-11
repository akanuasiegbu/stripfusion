# Strip-Fusion: Spatiotemporal Fusion for Multispectral Pedestrian Detection
Asiegbu Miracle Kanu-Asiegbu, Nitin Jotwani, and Xiaoxiao Du

Multimedia is in `multimedia_attachment` folder

## Steps for Training/Inference
1. **Build Docker Image**
   ```bash
   cd docker
   bash build.sh
   ```
   *Note: Make sure to change `USERNAME` and `USER_UID` in the script.*

2. **Run Docker Container**
   ```bash
   cd ..
   bash docker/run.sh
   ```
   *Note: Update `your_username` and `path_to_project_directory` accordingly.*


3. **Download and Prepare Dataset**
   - Download the KAIST-CVPR15 dataset from [here](https://soonminhwang.github.io/rgbt-ped-detection/) and place it in your dataset directory. (Kaist and CVC-14 is already downloaded and prepared )
   - Copy the dataset:
     ```bash
     cp -r ~/dataset/* path_to_your_dataset_directory
     ```
   - Copy the sanitized annotations format:
     ```bash
     cp sanitized_annotations_format_all path_to_your_dataset_directory
     ```

4. **Update Annotation Path**
   - Modify `KAIST_ANNOTATION_PATH` in `utils/datasets_vid.py` to the absolute path of `sanitized_annotations_format_all`. (Already Done)

5. **Training**
Important parameters: `--kl_cross`, `--n_roi`. Default is `--n_roi 300`, meaning kl diverge is performed as in paper shown. When `--n_roi 0`, kl divergence is not used. When `--n_roi 300` and `--kl_cross`, cross KL divergence is used. 

`--freeze_bb_rgb_bb_ir_det_rgb` is used to say that backbone is frozen while the rgb head is frozen. While `--freeze_bb_rgb_bb_ir_det_ir` is used to say that backbone is frozen while the thermal head is frozen. If `--thermal_weights` and `--rgb_weights` are loaded incorrectly this is not true.

To Train without using tadaconv need `--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolo_without_tadaconv.yaml `

Kaist Sample Training Code 
   ```
   python train_video.py \
   --data ./data/multispectral_temporal/kaist_video_sanitized_nc1_whole_updated.yaml \
   --batch-size 6 \
   --epochs 25 \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --dataset_used kaist \
   --regex_search '.set...V...' \
   --img-size 640 \
   --name select_useful_name  \
   --use_tadaconv \
   --detection_head lastframe \
   --hyp data/hyp.finetune_focal_loss_high_obj_low_scale.yaml \
   --sanitized \
   --temporal_mosaic \
   --mosaic \
   --save_all_model_epochs \
   --ignore_high_occ \
   --all_objects \
   --use_both_labels_for_optimization \
   --detector_weights both \
   --use_mode_spec_back_weights \
   --thermal_weights yolov5l_kaist_best_thermal.pt \
   --rgb_weights yolov5l_kaist_best_rgb.pt \
   --freeze_model \
   --freeze_bb_rgb_bb_ir_det_rgb
   ```
*Note: For Kaist Training, Update `--name select_useful_name` and `--batch_size 6` accordingly*
*Sanity check Transferred 1284/4876 items from yolov5l_kaist_best_rgb.pt and yolov5l_kaist_best_thermal.pt
Froze 522/4876 items from yolov5l_kaist_best_rgb.pt and yolov5l_kaist_best_thermal.pt *

CVC14 Sample Training Code, If want to use separate labels for training:`cvc14_align_resizedv2`, we use only rgb labels to train each head:`cvc14_align_resizedv2_use_only_rgb_anns` which uses only rgb labels
   ```
python train_video.py \
--data ./data/multispectral_temporal/cvc14_video_aligned_resizedv2_whole.yaml \
--batch-size 6 \
--epochs 40 \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--lframe 3 \
--temporal_stride 3 \
--gframe 0 \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--regex_search '.set...V...' \
--img-size 640 \
--name select_useful_name \
--temporal_mosaic \
--mosaic \
--use_tadaconv \
--hyp data/hyp.finetune_focal_loss_high_obj_low_scale.yaml \
--save_all_model_epochs \
--resize_cvc14_for_eval \
--detection_head lastframe \
--use_both_labels_for_optimization \
--detector_weights both \
--use_mode_spec_back_weights \
--thermal_weights yolov5l_kaist_best_thermal.pt \
--rgb_weights yolov5l_kaist_best_rgb.pt \
--freeze_model \
--freeze_bb_rgb_bb_ir_det_ir
```
*Note: For CVC-14 Training. `--name` and `--batch_size` accordingly. Reduce batch_size to make fit on one A100 gpu. Probably `--batch_size 6`*

*Transferred 1284/4876 (with tadaconv) or 1284/1664 items from point_to_kaist_model
Froze 522/4876 items from point_to_kaist_model*

6. **Testing**

Use the testing to obtain the json file. 

  For postprocessing these parameters might are useful  a)`--use_rgb_inference`, b) `--use_thermal_inference`, c) `--pp_fusion_nms`, and d `--conf-thres`.

  Note that for postprocessing, currently need to set threshold inside `post_process_fusion.py` and change the thres for `fusion_predictions` function.
  1) To use only visible detection apply a) and d). 
  2) To use only thermal detections apply b) and d). 
  3) To use both detections with only NMS apply a), b), and d).
  4) To use both detections with postprocessing, apply a), b), c) and d).

Note when doing inference on a model that does not use tadaconv turn off  `--use_tadaconv` and update the `yolov_modules_to_select.yaml` file and set to `use_tadaconv: False`.

  Kaist Testing
   ```
   python  test_video.py \
   --weights point_to_kaist_model \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --name select_useful_name_make_name_same_as_training \
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
   --use_both_labels_for_optimization \
   --conf-thres 0.2 \
   --use_rgb_inference

   ```
*Note that can use `--multiple_outputs` to obtain json from all the epochs, for visualization and numerical results use `--conf-thres 0.2` so can see the higher confidence bounding boxes.

  CVC14 Testing
   ```
   python  test_video.py \
   --weights point_to_cvc14_model \
   --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name select_useful_name_make_name_same_as_training \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
   --img-size 640 \
   --save-json \
   --use_tadaconv \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --use_both_labels_for_optimization \
   --resize_cvc14_for_eval \
   --conf-thres 0.2 \
   --use_rgb_inference


   ```
*Note that can use `--multiple_outputs` to obtain json from all the epochs, for visualization and numerical results use `--conf-thres 0.2` so can see the higher confidence bounding *

7. **Numerical Results**

Get numerical results for `--conf-thres 0.2`

Kaist Numerical Results
  ```
    python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles JSON_FILE
  ```

CVC-14 Numerical Results
   ```
  python miss_rate_and_map/evaluation_script_cvc14.py \
    --annFile json_gt/cvc-14_test_tl.json \
    --rstFiles JSON_FILE
  ```

*Note that can use `--multiple_outputs` to obtain missrate.json and plot/look at results from all the epochs*

8. **Visualization**

If you turn off `--plot_rgb_image`, results will plot on thermal image

*Visualization for Kaist*
  ```
  python visualization_using_missrate.py \
--data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
--dataset_used  kaist \
--annFile miss_rate_and_map/KAIST_annotation.json \
--rstFiles JSON_FILE\
--name chose_useful_name_related_to_model_name \
--plot_detections \
--plot_rgb_image \
--plot_gt_on_top \
--exist-ok
  ```

*Visualization for CVC-14*
   ```
   python visualization_using_missrate.py \
    --data ./data/multispectral_temporal/cvc14_video_test.yaml \
    --dataset_used  cvc14 \
    --annFile json_gt/cvc-14_test_tl.json \
    --rstFiles JSON_FILE \
    --name chose_useful_name_related_to_model_name \
    --plot_detections \
    --plot_rgb_image \
    --plot_gt_on_top \
    --exist-ok
  ```
  
* Note that since we resize the CVC-14 to correct size, we use the orginal images `cvc14_video_test.yaml` instead of `cvc14_video_test_resizedv2.yaml` to plot images (this is not a mistake)

