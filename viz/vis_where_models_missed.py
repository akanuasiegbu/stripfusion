from miss_rate_and_map import evaluation_script_cvc14, evaluation_script
import cv2
import argparse
from pathlib import Path


from utils.general import increment_path
import yaml
from utils.plots import colors
import random
import numpy as np
import json


#######################################################
#this is redundant 
def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # print(tl)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # print(c1, c2)
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        # tf = max(tl - 1, 1)  # font thickness
        tf = 1
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1]+ 50), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def match_gt_id(gts_one_image, gt_id_to_plot, img, color=[0, 255, 0]):
  
    for gt in gts_one_image:
        if gt['id'] in gt_id_to_plot:
            bbox = gt['bbox']
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
            plot_one_box(bbox, img, color=color)

def init_gtIds(gts, s_dtm_up_to_ffpi):
    s_gt_id = set()
    for gt in gts:
        if not gt['ignore']:
            s_gt_id.add(gt['id'])

    return s_gt_id.intersection(s_dtm_up_to_ffpi)

def create_paths(rst_file, opt, only_plot_up_to_fppi):
    path = Path(rst_file)
    
    path_names_mod = ["missed_ours_heavy", "missed_ms_detr_heavy", "detected_both_heavy"]
    end_name_mod = 'to_fppi' if only_plot_up_to_fppi else 'enire_json'
    
    path_name = []
    for folder_name in path_names_mod:
        
        folder_name= f'{folder_name}_{end_name_mod}'
     
        path_name.append(increment_path(Path(opt.project) / opt.name /
                        path.name.split('.')[0] /folder_name, exist_ok=opt.exist_ok) ) # increment run
        path_name[-1].mkdir(parents=True, exist_ok=True)
    
    
    return path_name[0], path_name[1], path_name[2]

def get_image_path( opt, data, plot_rgb_image):
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)

    val_path = data['val_rgb'] if plot_rgb_image else data['val_ir']

    # Map the id to correct location to get image 
    file = open(val_path) # just cuz path is visible in multispectral 

    path_map = file.read().splitlines()
    end = 2252 if opt.dataset_used == 'kaist' else 1417
    assert len(path_map) == end

    path_annotate_hash = {}
    for i in range(0,end):
        path_annotate_hash[i] = path_map[i]
    
    return path_annotate_hash

def vis_missed_gt(rst_file, annfile, opt, data, plot_rgb_image=True, only_plot_up_to_fppi=False):
    annfile = 'miss_rate_and_map/KAIST_annotation.json'
    # rstfile_our_best = 'runs/test/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json'
    rstfile_msdetr = 'runs/state_of_arts/conf_point_1/MS-DETR_result_0.2.txt'
    # rstfile_msdetr = 'runs/test/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div/cur_24_predictions_ct2_thermal.json'
    # rstfile_msdetr = 'runs/state_of_arts/MS-DETR_result.txt'
    # rstfile_msdetr = "runs/state_of_arts/use_for_vis/MS-DETR_result_0.32.txt"
    # rstfile_msdetr = "runs/test/yolov5_cvc14_tada_stripmlpv2_kl_div/cur_39_predictions_ct2_rgb.json"

    # create dir to save images into    
    save_dir_ours, save_dir_ms_detr, save_dir_both = create_paths(rst_file,opt, only_plot_up_to_fppi)
    
    # Get Image Path Hash table
    path_annotate_hash = get_image_path( opt, data, plot_rgb_image)

    if opt.dataset_used == "kaist":
        eval_result_our_best = evaluation_script.evaluate(annfile, rst_file, dataset_used='kaist')
        eval_result_ms_detr = evaluation_script.evaluate(annfile, rstfile_msdetr, dataset_used='kaist')
    elif opt.dataset_used == "cvc14":
        eval_result_our_best = evaluation_script.evaluate(annfile, rst_file, dataset_used='cvc14')
        eval_result_ms_detr = evaluation_script.evaluate(annfile, rstfile_msdetr, dataset_used='cvc14')
        

    gts_our_best = eval_result_our_best['all']._gts
    # gts_ms_detr = eval_result_ms_detr['all']._gts # same thing 
    
    # Note that if no objects detected evalImg is None even if ground truth objects exist because 
    # of 'if len(dt) == 0: return None' in evaluation_script
    evalImgs_our_best = eval_result_our_best['all'].evalImgs
    evalImgs_ms_detr = eval_result_ms_detr['all'].evalImgs


    if only_plot_up_to_fppi:
        ind = eval_result_our_best['all'].help_plot['inds'][-1]
        dtm_up_to_ffpi = eval_result_our_best['all'].help_plot['dtm'][:,:ind].reshape(-1).astype(int)
    else:
        dtm_up_to_ffpi = eval_result_our_best['all'].help_plot['dtm'].reshape(-1).astype(int)

    s_ours_dtm_up_to_ffpi = set(dtm_up_to_ffpi) 
    
    #  need to remove zero, though it might correspond to false postive
    s_ours_dtm_up_to_ffpi.remove(0)
    
    if only_plot_up_to_fppi:
        ind = eval_result_ms_detr['all'].help_plot['inds'][-1]
        dtm_up_to_ffpi = eval_result_ms_detr['all'].help_plot['dtm'][:,:ind].reshape(-1).astype(int)
    else:
        dtm_up_to_ffpi = eval_result_ms_detr['all'].help_plot['dtm'].reshape(-1).astype(int)
   
    s_ms_detr_dtm_up_to_ffpi = set(dtm_up_to_ffpi)
    
    # need to remove zero, though it might correspond to false postive
    s_ms_detr_dtm_up_to_ffpi.remove(0)
    
    for image_num, (evalImg_our_best, evalImg_ms_detr) in enumerate(zip(evalImgs_our_best, evalImgs_ms_detr)):
        img_missed_ours = cv2.imread(path_annotate_hash[image_num])
        img_missed_ms_detr = cv2.imread(path_annotate_hash[image_num])
        img_detected_both = cv2.imread(path_annotate_hash[image_num])
        
        # init so its gt that  are not ignored
        # if look at evalImg and no detections all gt  are missing
        gt_for_image = gts_our_best[(image_num,1)]
        
        # Takes care of zero removed in up_to_fppi
        gt_id_our_best = init_gtIds(gt_for_image, s_ours_dtm_up_to_ffpi)
        gt_id_ms_detr = init_gtIds(gt_for_image, s_ms_detr_dtm_up_to_ffpi)
        
        
        # 1) Detected in ours, not detectd in ms_detr
        missed_gt_ms_detr = gt_id_our_best - gt_id_ms_detr
        
        # 2) detected in ms_detr, not detected in ours
        missed_gt_ours = gt_id_ms_detr - gt_id_our_best
        
        # 3) detected in both
        gt_both = gt_id_our_best.intersection(gt_id_ms_detr)
        
        
        # Need to plot, just want to show gt so its clear where missing is happening
        if missed_gt_ours != set():
            match_gt_id(gt_for_image, missed_gt_ours, img_missed_ours)
            save_path = f"{save_dir_ours}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
            cv2.imwrite(save_path, img_missed_ours)
            
        
        if missed_gt_ms_detr != set():
            match_gt_id(gt_for_image, missed_gt_ms_detr, img_missed_ms_detr)
            save_path = f"{save_dir_ms_detr}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
            cv2.imwrite(save_path, img_missed_ms_detr)

        
        if gt_both != set():
            match_gt_id(gt_for_image, gt_both, img_detected_both)
            save_path = f"{save_dir_both}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
            cv2.imwrite(save_path, img_detected_both)
    print('Done')
  
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='miss_rate_and_map/KAIST_annotation.json',
                        help='Please put the path of the annotation file. Only support json format.')
    parser.add_argument('--rstFiles', type=str, nargs='+', default=['evaluation_script/MLPD_result.json'],
                        help='Please put the path of the result file. Only support json, txt format.')
    parser.add_argument('--evalFig', type=str, default='KASIT_BENCHMARK.jpg',
                        help='Please put the output path of the Miss rate versus false positive per-image (FPPI) curve')
    parser.add_argument('--multiple_outputs', action='store_true', help='evaluate muliple json and save result')
    parser.add_argument('--data', type=str, default='./data/multispectral_temporal/kaist_video_test.yaml', help='*.data path')
    parser.add_argument('--project', default='runs/detect_from_json', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, cvc14,')
    parser.add_argument('--point_to_scaled_image', action='store_true' , help='point to original size images')
    parser.add_argument('--plot_detections',action='store_true' , help='plot detections')
    parser.add_argument('--plot_gt_on_top',action='store_true' , help='point ground truth on top of the images')
    parser.add_argument('--use_score',action='store_true' , help='point ground truth on top of the images')
    parser.add_argument('--plot_trajectory',action='store_true' , help='Plot Trajectory')
    parser.add_argument('--lframe', type=int, default=3, help='Number of Local Frames in Batch')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Local Frames in a batch are strided by this amount')
    parser.add_argument('--detection_head', type=str, default='lastframe', help='selects the detection head')
    parser.add_argument('--plot_rgb_image',action='store_true', help='Plot using thermal or rgb image')
    parser.add_argument('--only_plot_up_to_fppi',action='store_true', help='Plot up to FPPI used, if false plot entire json/txt')

    args = parser.parse_args()

    
    vis_missed_gt(
                    rst_file = args.rstFiles[0], 
                    annfile=args.annFile,
                    opt = args,
                    data = args.data, 
                    plot_rgb_image = args.plot_rgb_image, 
                    only_plot_up_to_fppi = args.only_plot_up_to_fppi
                  )