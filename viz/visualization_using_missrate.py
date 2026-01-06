# The point of this code is to be able to visualize what is ignored in ground truth and what is ignored in the detection
# Also want to be able to see what is counting as True postive for missrate to understand where focus needs to occur to fix results


from pathlib import Path
from miss_rate_and_map import evaluation_script #evaluation_script_cvc14#, evaluation_script
from utils.general import increment_path
import argparse
import yaml
import cv2
from utils.plots import colors
import random
import numpy as np
from utils.datasets_vid import LoadMultiModalImagesAndLabels
import json
# alternate = 1
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
        cv2.putText(im, label, (c1[0], c1[1]-10), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def use_ids_ignore_to_plot(anns, Ids_Ignore, img, color, use_score=False, line_thickness=3):
    # Plotting Detections 
    for ann in anns:
        bbox = ann['bbox']
        bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        cur_id = ann['id']
        # import pdb; pdb.set_trace()
        score = round(ann['score'],2) if use_score else None
        score = f"{score:.2f}"if score is not None else None
        if Ids_Ignore[cur_id]:
            #plotting ignored bbox
            plot_one_box(bbox, img, label=score,  color=color[0], line_thickness=line_thickness)
        else:
            plot_one_box(bbox, img, label=score,  color=color[0], line_thickness=line_thickness)
            
def filter_heavy_occluded_images(annfile, height_min=20, occlusion=2):
    image_ids_to_look = set()

    with open(annfile, 'r') as file:
        data = json.load(file)
        for ele in data['annotations']:
            if ele['height'] >= height_min and  ele['occlusion'] == occlusion:
                image_ids_to_look.add(ele['image_id'])
    return image_ids_to_look



def visalize_missrate(annfile, rstfile, opt, data, plot_rgb_image=True):

    # create dir to save images into    
    path = Path(rstfile)
    save_dir = increment_path(Path(opt.project) / opt.name / path.name.split('.')[0], exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)

    if opt.filter_occlusion:
        image_nums_to_filter = filter_heavy_occluded_images(annfile)
        save_dir_occl = increment_path(Path(opt.project) / opt.name / path.name.split('.')[-1] / 'occlusion', exist_ok=opt.exist_ok)
        save_dir_occl.mkdir(parents=True, exist_ok=True)

    

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
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

    if opt.plot_trajectory:
        # Need to make a list of list with trajectory saved in side the list 
        trajectory_paths = []
        if 'cvc14' in opt.dataset_used:
            for frame_num in range(0,end):
                nearby_strided_frames = LoadMultiModalImagesAndLabels.get_frames_tublet(
                    path_map[frame_num], opt.temporal_stride, opt.lframe, opt.dataset_used,
                    opt.detection_head, path_map, frame_num
                    )
                
                trajectory_paths.append(nearby_strided_frames)
        elif 'kaist' in opt.dataset_used:
            for frame_num in range(0,end):
                nearby_strided_frames = LoadMultiModalImagesAndLabels.get_frames_tublet(
                    path_map[frame_num], opt.temporal_stride, opt.lframe, opt.dataset_used,
                    pt.detection_head
                    )
                trajectory_paths.append(nearby_strided_frames)
        else:
            raise 'not cvc14 or kaist'

    if 'cvc14' in opt.dataset_used:
        # import pdb; pdb.set_trace()
        print('Indside here cvc14')
        eval_result = evaluation_script.evaluate(annfile, rstfile, dataset_used='cvc14')
    elif 'kaist' in opt.dataset_used:
        print("inside here Kaist")
        eval_result = evaluation_script.evaluate(annfile, rstfile, dataset_used='kaist')
    else:
        raise 'not cvc14 or kaist'

    gts = eval_result['all']._gts # dict with tuple pair (imagenumber, classid) example when using only pedestrains its (imagenumber, 1)
    dts = eval_result['all']._dts # dict with tuple pair (imagenumber, classid) example when using only pedestrains its (imagenumber, 1)
    evalImgs = eval_result['all'].evalImgs # list but indexed based on image number
    for image_num, evalImg in enumerate(evalImgs):
        if opt.plot_trajectory:
            imgs_list = []
            for img_path in trajectory_paths[image_num]:
                img_in_traj = cv2.imread(img_path)
                img_in_traj = cv2.resize(img_in_traj, (471,640)) if opt.dataset_used == 'cvc14' else img_in_traj
                imgs_list.append(img_in_traj)
        
        
        img = cv2.imread(path_annotate_hash[image_num])
        if evalImg is None: # Note that if no objects detected evalImg is None even if ground truth objects exist becuase of 'if len(dt) == 0: return None' in evaluation_script
            save_path = f"{save_dir}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
            print(save_path)
            
            if opt.plot_trajectory:
                imgs_list.append(img)
                Hori = np.concatenate(imgs_list, axis=1) 
                cv2.imwrite(save_path, Hori)
            else:
                cv2.imwrite(save_path, img)
            continue

        if opt.plot_detections:
            dtIds = evalImg['dtIds'] 
            dtMatches = evalImg['dtMatches'] # not sure how to use this, if 0 I think means did not match to any
            dtIgnore =  list(evalImg['dtIgnore'][0]) #if 1 means dt bbox is being ignored, each element corrsponds to dtIds being ignored in the index level or not
            dtIds_Ignore = dict(zip(dtIds, dtIgnore ))
            cur_dets = dts[(image_num, 1)] # this is a list
            # # Plotting Detections
            use_ids_ignore_to_plot(anns=cur_dets, Ids_Ignore=dtIds_Ignore, img=img,  color=[ [0, 255, 255], [30, 150, 255] ], use_score=opt.use_score) # yellow, orange
        
        if opt.plot_gt_on_top:
            gtIds = evalImg['gtIds']
            gtMatches = evalImg['gtMatches'] # not sure how to use this, if 0 looks like it means this is not matched 
            gtIgnore = list(evalImg['gtIgnore']) # if 1 means gt bbox is being ignored, each element corrsponds to gtIds being ignored or not
            gtIds_Ignore = dict(zip(gtIds, gtIgnore))
            cur_gts = gts[(image_num, 1)] # this is a list
            # # Plotting Detections
            use_ids_ignore_to_plot(anns=cur_gts, Ids_Ignore=gtIds_Ignore, img=img, color=[ [0, 255, 0], [150, 255, 170] ],
                                   line_thickness=2) #green, turquoie
        

        if opt.filter_occlusion and image_num in image_nums_to_filter:
            save_path = f"{save_dir_occl}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
        else:
            save_path = f"{save_dir}/{image_num}{'_rgb.jpg' if plot_rgb_image else '_ir.jpg'}"
            

        print(save_path)
        if opt.plot_trajectory:
            img = cv2.resize(img, (471,640)) if opt.dataset_used == 'cvc14' else img
            imgs_list.append(img)
            Hori = np.concatenate(imgs_list, axis=1) 
            cv2.imwrite(save_path, Hori)
        else:
            cv2.imwrite(save_path, img)

            
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='evaluation_script/KAIST_annotation.json',
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
    parser.add_argument('--filter_occlusion',action='store_true', help='Plot using thermal or rgb image')




    args = parser.parse_args()

    print('#'*50)
    print(args.rstFiles[0])
    print('#'*50)

    visalize_missrate(annfile=args.annFile, rstfile=args.rstFiles[0], opt=args, data=args.data, plot_rgb_image=args.plot_rgb_image)