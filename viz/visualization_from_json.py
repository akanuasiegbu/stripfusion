import json
import argparse
import json
import os
from pathlib import Path
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy
import numpy as np
import yaml
import cv2
from utils.plots import colors, plot_one_box
import xml.etree.ElementTree as ET 
from utils.general import bbox_iou
import torch


def determine_bbox_then_plot(json_data, cur_i, img, color_det, gt_xml = None, use_intersects=False, overlap_indices = []):
    bbox = json_data[cur_i]['bbox'].copy() #xy_top_left width_height
    bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]

    # gt_bbox = read_one_xml_file(gt_xml) if use_intersects else None #x1y1x2y2
    
    # if gt_bbox == None:
    #     intersects = False
    # else:
    #     intersects = is_there_intersection(torch.tensor(bbox), torch.tensor(gt_bbox)) if gt_bbox != [] else False
    # is_there_intersection
    if opt.plot_score:
        score = str(round(json_data[cur_i]['score'], 3))
    else:
        score = None
    
    # if gt_bbox is None:
    plot_one_box(bbox, img, label=score, color=color_det) #Detections
    # else:
    #     if intersects:
    #         plot_one_box(bbox, img, label=score, color=color_det)
    #         overlap_indices.append(cur_i)


def is_there_intersection(bbox, gt_bbox):
    ious = bbox_iou(bbox, gt_bbox)
    return np.any(ious.numpy()>0)
    

def plot_json_results_and_ground_truth(json_path, data, opt, use_intersects, dataset_used ='kaist', point_to_scaled_image=False):
    #Save Directory
    path = Path(json_path)
    save_dir = increment_path(Path(opt.project) / opt.name / path.name.split('.')[0], exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    # Import the JSON file
    f = open(json_path)
    json_data = json.load(f)

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)

    
    val_path_rgb = data['val_rgb']
    val_path_ir = data['val_ir']
    # Map the id to correct location to get image 
    file = open(val_path_rgb) # just cuz path is visible in multispectral 
    path_map = file.read().splitlines()

    end = 2252 if dataset_used == 'kaist' else 1417

    path_annotate_hash = {}
    for i in range(0,end):
        path_annotate_hash[i] = {}
    
    for i in range( 0, end):
        img_path = path_map[i]
        if 'cvc14' in dataset_used:
            if point_to_scaled_image and 'Day' in img_path.split('/')[5]:
                img_path = img_path.replace('/Day/', '/Day_Resized/')
            elif point_to_scaled_image and 'Night' in img_path.split('/')[5]:
                img_path = img_path.replace('/Night/', '/Night_Resized/')

        path_annotate_hash[i] = {}
        path_annotate_hash[i]['img_path_rgb'] = img_path

        if dataset_used == 'kaist':
            xml_file = img_path.replace('images', 'annotations-xml-new-sanitized')
            xml_file = xml_file.replace('.jpg', '.xml')
            xml_file = xml_file.replace('/visible/', '/')

            path_annotate_hash[i]['xml_file'] = xml_file


    # Plot the images
    print('here')
    cur_image_id = json_data[0]['image_id']
    saved_images = []
    cur_i = 0
    
    next_image_id = json_data[cur_i+1]['image_id']
    color_det = [30, 255, 255]
    overlap_indices = []
    while cur_i < len(json_data) -1:
        img_path_rgb = path_annotate_hash[cur_image_id]['img_path_rgb']
        if dataset_used == 'kaist':
            xml_file = path_annotate_hash[cur_image_id]['xml_file'] #also this is the non-santinzied one
        else:
            xml_file=None
        img_rgb = cv2.imread(img_path_rgb)
        while next_image_id == cur_image_id:
            

            determine_bbox_then_plot(json_data, cur_i, img_rgb, color_det, gt_xml=xml_file,
                                     use_intersects=use_intersects, overlap_indices=overlap_indices)
            
            cur_i += 1
            cur_image_id = json_data[cur_i]['image_id']
            next_image_id = json_data[cur_i+1]['image_id'] if cur_i < len(json_data) -1 else 2252
            
        if next_image_id != cur_image_id:
            determine_bbox_then_plot(json_data, cur_i, img_rgb, color_det, gt_xml=xml_file,
                                        use_intersects=use_intersects, overlap_indices=overlap_indices)

            save_path_rgb = str(save_dir) + '/' + str(cur_image_id)+ '_rgb'+'.jpg'
            cv2.imwrite(save_path_rgb, img_rgb)
        
        cur_i += 1
        cur_image_id = json_data[cur_i]['image_id']
        if dataset_used == 'kaist':
            next_image_id = json_data[cur_i+1]['image_id'] if cur_i < len(json_data) -1 else 2252
        else:
            next_image_id = json_data[cur_i+1]['image_id'] if cur_i < len(json_data) -1 else 1417

        if use_intersects:
            write_json_to = str(save_dir / "detection_overlap_gt.json")  # predictions json
            overlap_json = []
            for i in overlap_indices:
                overlap_json.append(json_data[i])

            with open(write_json_to, 'w') as f:
                json.dump(overlap_json, f)
    return 
            

def read_one_xml_file(xmlfilename):
    tree = ET.parse(xmlfilename)
    root = tree.getroot()
    out = []
    class_map={'person':0, 'people':1, 'cyclist':2, 'person?':3}
    for obj in root.iter('object'):
        obj_type = obj.find('name').text
        
        x = float(obj.find('bndbox').find('x').text)
        y = float(obj.find('bndbox').find('y').text)
        w = float(obj.find('bndbox').find('w').text)
        h = float(obj.find('bndbox').find('h').text)

        # since kaist dataset was in top left, width height converted it to mid_X_Y w_h
        # if norm:
        #     out.append([class_map[obj_type], (x+w/2)/img_width, (y+h/2)/img_height, w/img_width, h/img_height ])
        # else:
        #     out.append([class_map[obj_type], x+w/2, y+h/2, w, h ])
        out.append([x,y, x +w, y+h])
    
    return out

def read_x1_x2_y1_y2_from_txt_annotations(path,name=''):
    #path='./CVC14/Night/Visible/Train/Annotations/2014_05_04_23_16_06_620000.txt'
    bbox_info=[]#最后返回的坐标
    with open(path,'r' ) as day_visible_f:
        day_visible_str = day_visible_f.read()  #可以是随便对文件的操作
        if day_visible_str=='':
            pass
            #print( "--------------" +path+ name + "is null--------------")
        else:
            day_visible_line_all=day_visible_str.split('\n')
            day_visible_line_i = 0 #计数
            for day_visible_line in day_visible_line_all:
                if day_visible_line=='':
                    break
                # bbox_info_one = [0, 0, 0, 0,0]
                bbox_info_one = [0, 0, 0, 0]
                day_visible_line_i=day_visible_line_i+1
                #print(path)
                #print(day_visible_line)
                x, y, w1, h1, one, zero1, zero2, zero3, zero4, num, zero5 = day_visible_line.split(' ')
                pass
                #处理异常情况
                # if(one != '1' or zero1 != '0' or zero2 != '0'  or  zero3 != '0' or zero4 != '0' or num != str(day_visible_line_i) or zero5 != '0'):
                # if(one != '1' or zero1 != '0' or zero2 != '0'  or  zero3 != '0' or zero4 != '0'  or zero5 != '0'):
                #     print(path+ name+" one,zero1, zero2, zero3, zero4, num, zero5 ",one,zero1, zero2, zero3, zero4, num, zero5 )
                x=int(x)-1
                y=int(y)-1
                w1= int(w1)
                h1= int(h1)

                x1= int(x-w1/2)
                y1= int(y-h1/2)
                x2 = int(x+w1/2)
                y2 =int(y+h1/2)
                # 坐标处理异常情况
                if x1<0:
                    #print( path + name + " x1<0 " ,x1)
                    x1=0
                if y1<0:
                    #print( path + name + " y1<0 " ,y1)
                    y1=0
                if x2>639:
                    #print( path + name + " x2>639 " ,x2)
                    x2=639
                if y2>470:
                    #print( path + name + " y2>470 " ,y2)
                    y2=470

                if(x1==0 and x2==0 and y1==0 and y2==0):
                    pass
                bbox_info_one[0] = x1
                bbox_info_one[1] = y1
                bbox_info_one[2] = w1
                bbox_info_one[3] = h1
                # bbox_info_one[4] = 0 # 0 means only using person
                bbox_info.append(bbox_info_one)
    return bbox_info
                        
def plot_ground_truth(json_path, data, dataset_used, overlay=True):
    #Save Directory
    path = Path(json_path)
    if overlay:
        save_dir = str(Path(opt.project) / opt.name / path.name.split('.')[0])
    else: 
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)
    # Import the JSON file
    f = open(json_path)
    # json_data = json.load(f)

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)

    color_gt = (0,255,0)
    val_path_rgb = data['val_rgb']
    val_path_ir = data['val_ir']
    # Map the id to correct location to get image 
    file = open(val_path_rgb) # just cuz path is visible in multispectral 
    paths = file.read().splitlines()
    if dataset_used == 'kaist':
        index = np.arange(0,2252)
    elif dataset_used == 'cvc14':
        index = np.arange(0,1417)

    path_map = dict(zip(index, paths))

    if overlay:
        images_in_dir = os.listdir(save_dir)
        images_in_dir = set(images_in_dir)

    for cur_image_id in index:

        # /mnt/workspace/datasets/kaist-cvpr15/images/set06/V000/visible/I00039.jpg
        if overlay:
            im = f'{cur_image_id}_rgb.jpg'
            img_pathorg = path_map[cur_image_id]
            img_path = save_dir + '/' + f'{cur_image_id}_rgb.jpg'
            os.makedirs(f'{save_dir}_with_gt', exist_ok=True)
            img_path_to_write = f'{save_dir}_with_gt/{cur_image_id}_rgb.jpg'
            in_dir = im in images_in_dir
            if in_dir:
                img_rgb = cv2.imread(img_path)
            else:
                img_rgb = cv2.imread(img_pathorg)

        else:
            img_pathorg = path_map[cur_image_id]
            img_rgb = cv2.imread(img_pathorg)
        
        if dataset_used == 'kaist':
            xml_file = img_pathorg.replace('images', 'annotations-xml-new-sanitized')
            xml_file = xml_file.replace('.jpg', '.xml')
            xml_file = xml_file.replace('/visible/', '/')
            bboxes = read_one_xml_file(xml_file) #these are the non-sanitized annotions 
        elif dataset_used =='cvc14':
            txt_file = img_pathorg.replace('FramesPos', 'Annotations').replace('.tif', '.txt')
            bboxes = read_x1_x2_y1_y2_from_txt_annotations(txt_file)
        
        for bbox in bboxes:
            plot_one_box(bbox, img_rgb, color=color_gt) #ground truth
        
        if overlay:
            cv2.imwrite(img_path_to_write, img_rgb)
        else:
            save_path_rgb = str(save_dir) + '/' + str(cur_image_id)+ '_rgb'+'.jpg'
            cv2.imwrite(save_path_rgb, img_rgb)
        
# 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='./data/multispectral_temporal/kaist_video_test.yaml', help='*.data path')
    parser.add_argument('--project', default='runs/detect_from_json', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel,')
    parser.add_argument('--json_file', type=str, default='./runs/test/fusion_transformerx3_kaist_video111_lframe_3_stride_1/best_predictions.json' , help='json file generated from test_video.py')
    parser.add_argument('--plot-score', action='store_true' , help='plot confidence values on images')
    parser.add_argument('--point_to_scaled_image', action='store_true' , help='point to original size images')
    parser.add_argument('--plot_gt_on_top',action='store_true' , help='point ground truth on top of the images')
    opt = parser.parse_args()
    
    
    if not opt.plot_gt_on_top:
        plot_json_results_and_ground_truth(opt.json_file, opt.data, opt, use_intersects=False, dataset_used=opt.dataset_used, point_to_scaled_image=opt.point_to_scaled_image)
    else:
        plot_ground_truth(opt.json_file, opt.data, dataset_used=opt.dataset_used, overlay=True) #Overlays groundtruth on detections, (not efficent but works)