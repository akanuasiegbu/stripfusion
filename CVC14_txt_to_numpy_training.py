#Obtain testing data from https://github.com/CalayZhou/MBNet/issues/28
# https://github.com/CalayZhou/MBNet/files/5352777/CVC14_txt_to_numpy.txt

#coding=utf-8
import numpy as np
import os
from os.path import join
import argparse
import yaml
from utils.datasets_vid import create_dataloader_rgb_ir
from utils.general import check_img_size, colorstr
import re
import json

#kaist_example = np.load("22158train_data_all4.npy")
#print("done!!")
###################
#训练集标签制作
###################
image_train_num_txthave_read = 0# The total number of txt tags read
image_train_num_none = 0 # Label is empty count
image_train_lens_notequal_none = 0 #The number of pedestrians in infrared and RGB is different
image_train_num=0 # Total number of training images
image_train_person_num = 0 #Total number of training pedestrians
image_train_list=[] #The last training list returned

pedestrian_aligned_num = 0 #  Align the number of pedestrians
pedestrian_notaligned_num = 0  # The number of pedestrians is not aligned


def read_x1_x2_y1_y2_from_txt_annotations(path,name='', use_midxy_wh=True, img_width=640, img_height=471):
    #path='./CVC14/Night/Visible/Train/Annotations/2014_05_04_23_16_06_620000.txt'
    bbox_info=[]# Last returned coordinates
    with open(path,'r' ) as day_visible_f:
        day_visible_str = day_visible_f.read()  
        if day_visible_str=='':
            pass
            #print( "--------------" +path+ name + "is null--------------")
        else:
            day_visible_line_all=day_visible_str.split('\n')
            day_visible_line_i = 0 
            for day_visible_line in day_visible_line_all:
                if day_visible_line=='': #handles empty lines
                    break
                bbox_info_one = [0, 0, 0, 0,0]
                day_visible_line_i=day_visible_line_i+1
                #print(path)
                #print(day_visible_line)
                out_annotations = day_visible_line.split(' ')
                if len(out_annotations) == 11:
                    x, y, w1, h1, one, zero1, zero2, zero3, zero4, num, zero5 = out_annotations
                else:
                    # useful for training data as some have extra space, for example './CVC-14/Night/Visible/Train/Annotations/2014_05_04_23_16_06_620000.txt'
                    # len(day_visible_line_all) counts empty lines which would be skipped 

                    # print('file {} has {} annotations'.format(path, str(len(day_visible_line_all))))
                    annotations_no_space = []
                    for ann in out_annotations:
                        if ann != '':
                            annotations_no_space.append(ann)
                    x, y, w1, h1, one, zero1, zero2, zero3, zero4, num, zero5 = annotations_no_space
                    
                # x, y, w1, h1, one, zero1, zero2, zero3, zero4, num, zero5 = day_visible_line.split(' ')
                # pass
                # Handle exceptions
                # if(one != '1' or zero1 != '0' or zero2 != '0'  or  zero3 != '0' or zero4 != '0' or num != str(day_visible_line_i) or zero5 != '0'):
                if(one != '1' or zero1 != '0' or zero2 != '0'  or  zero3 != '0' or zero4 != '0'  or zero5 != '0'):
                    print(path+ name+" one,zero1, zero2, zero3, zero4, num, zero5 ",one,zero1, zero2, zero3, zero4, num, zero5 )
                x=int(x)-1
                y=int(y)-1
                w1= int(w1)
                h1= int(h1)
                if not use_midxy_wh:
                    #x1= int(x-w1/2)
                    #y1= int(y-h1/2)
                    #x2 = int(x+w1/2)
                    #y2 =int(y+h1/2)
                    x1= x-w1/2
                    y1= y-h1/2
                    x2 = x+w1/2
                    y2 =y+h1/2
                    # Coordinate handling exceptions
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
                    bbox_info_one[2] = x2
                    bbox_info_one[3] = y2
                    bbox_info_one[4] = int(num)#Save label
                    bbox_info.append(bbox_info_one)
                else:
                    bbox_info_one[0] = x/img_width
                    bbox_info_one[1] = y/img_height
                    bbox_info_one[2] = w1/img_width
                    bbox_info_one[3] = h1/img_height
                    bbox_info_one[4] = 0 # 0 means only using person
                    bbox_info.append(bbox_info_one)
    return bbox_info
def bbox_info_combine_aligned(bbox_info_night_visible,bbox_info_night_FIR, scale_to_640_512=False, img_width=640, img_height=471, pedestrian_aligned_num=0, pedestrian_notaligned_num=0, just_one_label=0,  just_one_label_ir=0 ,just_one_label_rgb=0 ):
    rgb_labels = []
    ir_labels = []
    if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
        bbox_info_large=bbox_info_night_visible
        bbox_info_small=bbox_info_night_FIR
        pedestrian_num=len(bbox_info_night_visible)
    elif(len(bbox_info_night_visible)<len(bbox_info_night_FIR)):
        bbox_info_small=bbox_info_night_visible
        bbox_info_large=bbox_info_night_FIR
        pedestrian_num = len(bbox_info_night_FIR)
    else:
        bbox_info_small=bbox_info_night_visible
        bbox_info_large=bbox_info_night_FIR
        pedestrian_num = len(bbox_info_night_FIR)
        
    id_large_dict,id_small_dict  = {}, {}
    for bbox_i in range(pedestrian_num):
        id_large_dict[bbox_info_large[bbox_i][4]] = bbox_i
    for bbox_j in range(len(bbox_info_small)):
        id_small_dict[bbox_info_small[bbox_j][4]] = bbox_j
    
    for bbox_i in range(pedestrian_num):
        ped_id = bbox_info_large[bbox_i][4]
        if ped_id in id_small_dict:
            bbox_j = id_small_dict[ped_id]

            x1_large = bbox_info_large[bbox_i][0]
            y1_large = bbox_info_large[bbox_i][1] if not scale_to_640_512 else bbox_info_large[bbox_i][1]*1.087044832
            x2_large = bbox_info_large[bbox_i][2]
            y2_large = bbox_info_large[bbox_i][3] if not scale_to_640_512 else bbox_info_large[bbox_i][3]*1.087044832

            x1_small = bbox_info_small[bbox_j][0]
            y1_small = bbox_info_small[bbox_j][1] if not scale_to_640_512 else bbox_info_small[bbox_j][1]*1.087044832
            x2_small = bbox_info_small[bbox_j][2]
            y2_small = bbox_info_small[bbox_j][3] if not scale_to_640_512 else bbox_info_small[bbox_j][3]*1.087044832

            pedestrian_aligned_num = pedestrian_aligned_num + 1
            w_large = x2_large - x1_large
            h_large = y2_large - y1_large
            w_small = x2_small - x1_small
            h_small = y2_small - y1_small
            if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
                # bbox_info_large=bbox_info_night_visible
                # bbox_info_small=bbox_info_night_FIR
                rgb_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
                ir_labels.append([(x1_small + w_small/2)/img_width, (y1_small+h_small/2)/img_height, w_small/img_width, h_small/img_height])
            else:
                # bbox_info_small=bbox_info_night_visible
                # bbox_info_large=bbox_info_night_FIR
                pedestrian_num = len(bbox_info_night_FIR)
                rgb_labels.append([(x1_small + w_small/2)/img_width, (y1_small+h_small/2)/img_height, w_small/img_width, h_small/img_height])
                ir_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
  
    
    
    if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
        just_one_label_rgb = just_one_label_rgb + len(id_large_dict) -len(rgb_labels)
        just_one_label_ir = just_one_label_ir + len(id_small_dict) -len(ir_labels)
    else:
        just_one_label_rgb = just_one_label_rgb + len(id_small_dict) -len(rgb_labels)
        just_one_label_ir = just_one_label_ir + len(id_large_dict) -len(ir_labels)
    
    
    return rgb_labels, ir_labels, pedestrian_aligned_num, pedestrian_notaligned_num, just_one_label, just_one_label_ir, just_one_label_rgb

def bbox_info_combine_alignedv2(bbox_info_night_visible,bbox_info_night_FIR, scale_to_640_512=False, img_width=640, img_height=471, pedestrian_aligned_num=0, pedestrian_notaligned_num=0, just_one_label=0, just_one_label_ir=0 ,just_one_label_rgb=0):
    rgb_labels = []
    ir_labels = []
    if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
        bbox_info_large=bbox_info_night_visible
        bbox_info_small=bbox_info_night_FIR
        pedestrian_num=len(bbox_info_night_visible)
    elif(len(bbox_info_night_visible)<len(bbox_info_night_FIR)):
        bbox_info_small=bbox_info_night_visible
        bbox_info_large=bbox_info_night_FIR
        pedestrian_num = len(bbox_info_night_FIR)
    else:
        bbox_info_small=bbox_info_night_visible
        bbox_info_large=bbox_info_night_FIR
        pedestrian_num = len(bbox_info_night_FIR)

    id_large_dict,id_small_dict  = {}, {}
    for bbox_i in range(pedestrian_num):
        id_large_dict[bbox_info_large[bbox_i][4]] = bbox_i
    for bbox_j in range(len(bbox_info_small)):
        id_small_dict[bbox_info_small[bbox_j][4]] = bbox_j
    
    if len(bbox_info_small)==0 and  len(bbox_info_night_visible)!=len(bbox_info_night_FIR): 
        for bbox_i in range(pedestrian_num):
            # pedestrian_aligned_num = pedestrian_aligned_num + 1
            just_one_label = just_one_label +1
            x1_large = bbox_info_large[bbox_i][0]
            y1_large = bbox_info_large[bbox_i][1] if not scale_to_640_512 else bbox_info_large[bbox_i][1]*1.087044832
            x2_large = bbox_info_large[bbox_i][2]
            y2_large = bbox_info_large[bbox_i][3] if not scale_to_640_512 else bbox_info_large[bbox_i][3]*1.087044832
            w_large = x2_large - x1_large
            h_large = y2_large - y1_large
            if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
                # bbox_info_large=bbox_info_night_visible
                rgb_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
            elif(len(bbox_info_night_visible)<len(bbox_info_night_FIR)):
                # bbox_info_large=bbox_info_night_FIR
                ir_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
            else:
                raise 'Shouldnt be here'
    else:
        for bbox_i in range(pedestrian_num):
            ped_id = bbox_info_large[bbox_i][4]
            if ped_id in id_small_dict:
                bbox_j = id_small_dict[ped_id]

                x1_large = bbox_info_large[bbox_i][0]
                y1_large = bbox_info_large[bbox_i][1] if not scale_to_640_512 else bbox_info_large[bbox_i][1]*1.087044832
                x2_large = bbox_info_large[bbox_i][2]
                y2_large = bbox_info_large[bbox_i][3] if not scale_to_640_512 else bbox_info_large[bbox_i][3]*1.087044832

                x1_small = bbox_info_small[bbox_j][0]
                y1_small = bbox_info_small[bbox_j][1] if not scale_to_640_512 else bbox_info_small[bbox_j][1]*1.087044832
                x2_small = bbox_info_small[bbox_j][2]
                y2_small = bbox_info_small[bbox_j][3] if not scale_to_640_512 else bbox_info_small[bbox_j][3]*1.087044832

                # 计算iou
                x_min_out = min(x1_large, x1_small)
                x_min_in = max(x1_large, x1_small)
                y_min_out = min(y1_large, y1_small)
                y_min_in = max(y1_large, y1_small)

                x_max_out = max(x2_large, x2_small)
                x_max_in = min(x2_large, x2_small)
                y_max_out = max(y2_large, y2_small)
                y_max_in = min(y2_large, y2_small)

                Iou_Large = (x_max_out - x_min_out) * (y_max_out - y_min_out)
                Iou_Small = (x_max_in - x_min_in) * (y_max_in - y_min_in)
                Iou = float(Iou_Small) / float(Iou_Large)
                if (Iou < 0.5):
                        pedestrian_notaligned_num = pedestrian_notaligned_num + 1
                else:
                    pedestrian_aligned_num = pedestrian_aligned_num + 1
                    w_large = x2_large - x1_large
                    h_large = y2_large - y1_large
                    w_small = x2_small - x1_small
                    h_small = y2_small - y1_small
                    if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
                        # bbox_info_large=bbox_info_night_visible
                        # bbox_info_small=bbox_info_night_FIR
                        rgb_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
                        ir_labels.append([(x1_small + w_small/2)/img_width, (y1_small+h_small/2)/img_height, w_small/img_width, h_small/img_height])
                    else:
                        # bbox_info_small=bbox_info_night_visible
                        # bbox_info_large=bbox_info_night_FIR
                        pedestrian_num = len(bbox_info_night_FIR)
                        rgb_labels.append([(x1_small + w_small/2)/img_width, (y1_small+h_small/2)/img_height, w_small/img_width, h_small/img_height])
                        ir_labels.append([(x1_large+w_large/2)/img_width, (y1_large+h_large/2)/img_height, w_large/img_width, h_large/img_height])
            
    if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
        just_one_label_rgb = just_one_label_rgb + len(id_large_dict) -len(rgb_labels)
        just_one_label_ir = just_one_label_ir + len(id_small_dict) -len(ir_labels)
    else:
        just_one_label_rgb = just_one_label_rgb + len(id_small_dict) -len(rgb_labels)
        just_one_label_ir = just_one_label_ir + len(id_large_dict) -len(ir_labels)

    return rgb_labels, ir_labels, pedestrian_aligned_num, pedestrian_notaligned_num, just_one_label, just_one_label_ir, just_one_label_rgb

def create_save_txtfolder(timeofday, out_dir, train_or_test, scale_to_640_512=False ):
    """
    timeofday: either 'Day' or 'Night'
    train_or_test: either 'Train'
    """
    os.makedirs( join(out_dir, 'labels_cvc14'), exist_ok=True)
    # os.makedirs( join(join(out_dir, 'labels_cvc14'),  '{}_{}'.format(timeofday,'visible')), exist_ok=True)
    # os.makedirs( join(join(out_dir, 'labels_cvc14'),  '{}_{}'.format(timeofday,'FIR')), exist_ok=True)

    _timeofday = timeofday if not scale_to_640_512 else f'{timeofday}_Resized'
    os.makedirs( join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'Visible', train_or_test)), exist_ok=True)
    os.makedirs( join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'FIR', train_or_test)), exist_ok=True)
    

def save_data(path_CVC14_train_visible, path_CVC14_train_FIR, timeofday, train_or_test, scale_to_640_512=False, out_dir=os.getcwd()):
    """
    path_CVC14_train_visible: CVC14 train visible  with timeofday
    path_CVC14_train_FIR: CVC14 train thermal with timeofday
    timeofday: either 'Day' or 'Night'
    train_or_test: either 'Train' or 'NewTest'
    out_dir: directory to save to
    """
    assert train_or_test == 'Train', 'make sure to spell correctly'
    create_save_txtfolder(timeofday, out_dir, train_or_test, scale_to_640_512=scale_to_640_512)
    pedestrian_aligned_num=0
    pedestrian_notaligned_num=0 
    just_one_label = 0
    just_one_label_ir = 0
    just_one_label_rgb = 0
    # For visible
    visible_files = os.listdir(path_CVC14_train_visible)
    visible_files.sort()

    # For thermal
    thermal_files = os.listdir(path_CVC14_train_FIR)
    thermal_files.sort()

    assert len(thermal_files) == len(visible_files)
    for train_visible_i, train_fir_i in zip(visible_files, thermal_files):#[345:]:  # set
        # image_train_num_txthave_read = image_train_num_txthave_read + 1
        assert train_fir_i ==  train_visible_i
        path_visible_sample = path_CVC14_train_visible + '/' + train_visible_i
        path_fir_sample = path_CVC14_train_FIR + '/' + train_fir_i
        isdir = os.path.isdir(path_visible_sample)
        isdir_ir = os.path.isdir(path_fir_sample)
        if isdir and isdir_ir: continue
        bbox_info_visible = read_x1_x2_y1_y2_from_txt_annotations(path_visible_sample, name='{}_visible'.format(timeofday), use_midxy_wh=False)
        bbox_info_fir = read_x1_x2_y1_y2_from_txt_annotations(path_fir_sample, name='{}_FIR'.format(timeofday), use_midxy_wh=False)
        if not scale_to_640_512:
            rgb_labels, ir_labels, pedestrian_aligned_num, pedestrian_notaligned_num, just_one_label, just_one_label_ir, just_one_label_rgb= bbox_info_combine_aligned(bbox_info_visible,
                                                                                                                bbox_info_fir, 
                                                                                                                pedestrian_aligned_num=pedestrian_aligned_num, 
                                                                                                                pedestrian_notaligned_num=pedestrian_notaligned_num,
                                                                                                                just_one_label = just_one_label,
                                                                                                                just_one_label_ir = just_one_label_ir,
                                                                                                                just_one_label_rgb=just_one_label_rgb)
        else:
           rgb_labels, ir_labels, pedestrian_aligned_num, pedestrian_notaligned_num, just_one_label, just_one_label_ir, just_one_label_rgb= bbox_info_combine_aligned(bbox_info_visible,
                                                                                                                bbox_info_fir,
                                                                                                                scale_to_640_512=scale_to_640_512, 
                                                                                                                img_height=512,
                                                                                                                img_width=640,
                                                                                                                pedestrian_aligned_num=pedestrian_aligned_num, 
                                                                                                                pedestrian_notaligned_num=pedestrian_notaligned_num,
                                                                                                                just_one_label = just_one_label,
                                                                                                                just_one_label_ir = just_one_label_ir,
                                                                                                                just_one_label_rgb=just_one_label_rgb)
            
        # For Visible
        _timeofday = timeofday if not scale_to_640_512 else f'{timeofday}_Resized'
        if rgb_labels == []:
            #creates an empty file
            open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'Visible', train_or_test)), train_visible_i),'w').close()
        else:        
            for bbox in rgb_labels:
                x,y,w,h = bbox
                label = 0
                with open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'Visible', train_or_test)), train_visible_i),'a') as file: 
                    file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, x,y, w,h))
        
        # For thermal
        if ir_labels == []:
            open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'FIR', train_or_test)), train_fir_i),'w').close()
        else:
            for bbox in ir_labels:
                x,y,w,h = bbox
                label=0
                with open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'FIR', train_or_test)), train_fir_i),'a') as file:
                    file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, x,y, w,h))
        
    print(f'unaligned {pedestrian_notaligned_num} , aligned {pedestrian_aligned_num}, justonelabel {just_one_label}, justoneir {just_one_label_ir}, justonergb {just_one_label_rgb}')
        

    # for train_fir_i in thermal_files:  # set
    #     # image_train_num_txthave_read = image_train_num_txthave_read + 1
    #     if isdir: continue
    #     if bbox_info_fir == []:
    #         open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(timeofday,'FIR', train_or_test)), train_fir_i),'w').close()
    #     else:
    #         for bbox in bbox_info_fir:
    #             x,y,w,h, label = bbox
    #             with open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(timeofday,'FIR', train_or_test)), train_fir_i),'a') as file:
    #                 file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, x,y, w,h))
        
        
def generate_image_location_txt(day_visible, night_visible, day_fir, night_fir, scale_to_640_512=False, train_or_test='Train', out_dir=os.getcwd()):
    """
    inputs:
        day_visible: location of visible images (use only positive frames as those are ones with labels and pedestrians)
        night_visible: location of visible images (use only positive frames as those are ones with labels and pedestrians)
        day_fir: location of thermal images (use only positive frames as those are ones with labels and pedestrians)
        night_fir: location of thermal images (use only positive frames as those are ones with labels and pedestrians)
        train_or_test: either 'Train'
    outputs: None
    """
    assert train_or_test == 'Train', 'make sure to spell correctly'
    # Need to seperate visible and thermal
    def full_path(image_names, absolute_path):
        image_location_absolute = []
        for image_name in image_names:
            image_location_absolute.append(join(absolute_path, image_name))
        return image_location_absolute
    
    #############################################################################################
    visible_set = set()
    fir_set = set()
    # Visible:
    visible_files_day = os.listdir(day_visible)
    visible_files_night = os.listdir(night_visible)
    visible_files_day.sort()
    visible_files_night.sort()

    visible_files_day_absolute_path = full_path(visible_files_day, day_visible)
    visible_files_night_absolute_path = full_path(visible_files_night, night_visible)
    visible_files_day_absolute_path.extend(visible_files_night_absolute_path)
    
    for image_path in visible_files_day_absolute_path:
        if image_path[-3:] == 'tif': #removes
            visible_set.add(image_path.split('/')[-1].split('.')[0]) # for example add '2014_05_01_23_18_25_804000' filename
            with open(join(out_dir, f'labels_cvc14/cvc14_{train_or_test.lower()}_visible.txt'), 'a') as file:
                if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Day/', '/Day_Resized/')
                elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Night/', '/Night_Resized/')

                file.write('{}\n'.format(image_path))
    ###############################################################################################
    # Thermal
    fir_files_day = os.listdir(day_fir)
    fir_files_night = os.listdir(night_fir)
    fir_files_day.sort()
    fir_files_night.sort()

    fir_files_day_absolute_path = full_path(fir_files_day, day_fir)
    fir_files_night_absolute_path = full_path(fir_files_night, night_fir)
    fir_files_day_absolute_path.extend(fir_files_night_absolute_path)

    for image_path in fir_files_day_absolute_path:
        if image_path[-3:] == 'tif':
            fir_set.add(image_path.split('/')[-1].split('.')[0])
            # if train_or_test =="NewTest" and (fir_name not in visible_set): 
            #     print(f'images not in fir {fir_name}')
            #     continue

            with open(join(out_dir, f'labels_cvc14/cvc14_{train_or_test.lower()}_fir.txt'), 'a') as file:
                if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Day/', '/Day_Resized/')
                elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Night/', '/Night_Resized/')
                file.write('{}\n'.format(image_path))

    #checking temporal misalignment
    for image_path in fir_files_day_absolute_path:
        fir_name = image_path.split('/')[-1].split('.')[0]
        if fir_name not in visible_set:
            with open(join(out_dir, f'labels_cvc14/fir_files_not_in_visible_{train_or_test.lower()}.txt'), 'a') as file:
                if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Day/', '/Day_Resized/')
                elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Night/', '/Night_Resized/')
                file.write('{}\n'.format(image_path))

    for image_path in visible_files_day_absolute_path:
        visible_name = image_path.split('/')[-1].split('.')[0]
        if visible_name not in fir_set:
            with open(join(out_dir, f'labels_cvc14/visible_files_not_in_fir_{train_or_test.lower()}.txt'), 'a') as file:
                if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Day/', '/Day_Resized/')
                elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
                    image_path = image_path.replace('/Night/', '/Night_Resized/')
                file.write('{}\n'.format(image_path))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_fusion_add_FLIR_aligned.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--lframe', type=int, default=6, help='Number of Local Frames in Batch')
    parser.add_argument('--gframe', type=int, default=0, help='Number of Global Frames in Batch')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Local Frames in a batch are strided by this amount')
    parser.add_argument('--regex_search', type=str, default=".set...V...", help="For kaist:'.set...V...' , For camel use:'.images...' .This helps the dataloader seperate ordered list in indivual videos for kaist use:r'.set...V...' ")
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel, cvc14, cvc14_align')
    parser.add_argument('--temporal_mosaic', action='store_true', help='load mosaic with temporally related sequences of images')
    parser.add_argument('--use_tadaconv', action='store_true', help='load tadaconv as feature extractor')
    parser.add_argument('--image_set', type=str, default="test", help='train, val, test')
    parser.add_argument('--json_gt_loc', type=str, default='./json_gt/')
    parser.add_argument('--midframe', action='store_true')
    parser.add_argument('--mosaic', action='store_true', help='use mosaic augmentations')
    parser.add_argument('--sanitized', action='store_true', help='using sanitized label only')
    parser.add_argument('--detection_head', type=str, default='lastframe', choices=['lastframe', 'midframe', 'fullframes'], help='selects the detection head: 1) lastframe, 2) midframe, 3) fullframes')

    opt = parser.parse_args()

    train_or_test = 'Train' #'Train' or 'NewTest'
    generate_json_train = False
    image_path = '/mnt/workspace/datasets/CVC-14'
    scale_to_640_512 = False
    print(f'scale_to_640_512 is set to {scale_to_640_512}')
    
    if train_or_test == 'Train':
        path_CVC14_train_day_visible = "/mnt/workspace/datasets/CVC-14/Day/Visible/Train/Annotations"
        path_CVC14_train_day_FIR = "/mnt/workspace/datasets/CVC-14/Day/FIR/Train/Annotations"
        path_CVC14_train_night_visible = "/mnt/workspace/datasets/CVC-14/Night/Visible/Train/Annotations"
        path_CVC14_train_night_FIR = "/mnt/workspace/datasets/CVC-14/Night/FIR/Train/Annotations"

        save_data(path_CVC14_train_visible=path_CVC14_train_day_visible, 
                        path_CVC14_train_FIR=path_CVC14_train_day_FIR,
                        scale_to_640_512=scale_to_640_512,
                        timeofday='Day',
                        train_or_test='Train')
        

        save_data(path_CVC14_train_visible=path_CVC14_train_night_visible,
                        path_CVC14_train_FIR=path_CVC14_train_night_FIR,
                        scale_to_640_512=scale_to_640_512,
                        timeofday='Night',
                        train_or_test='Train')
        framespos_night_fir ='/mnt/workspace/datasets/CVC-14/Night/FIR/Train/FramesPos'
        framespos_night_visible='/mnt/workspace/datasets/CVC-14/Night/Visible/Train/FramesPos'
        framespos_day_fir ='/mnt/workspace/datasets/CVC-14/Day/FIR/Train/FramesPos'
        framespos_day_visible='/mnt/workspace/datasets/CVC-14/Day/Visible/Train/FramesPos'
        generate_image_location_txt(day_visible=framespos_day_visible, 
                                             night_visible=framespos_night_visible, 
                                             day_fir=framespos_day_fir, 
                                             night_fir=framespos_night_fir,
                                             scale_to_640_512=scale_to_640_512,
                                             train_or_test='Train')
    
    if generate_json_train:
        
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps
        
        opt.hyp = hyp
            
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict
        train_path_rgb = data_dict['train_rgb']
        test_path_rgb = data_dict['val_rgb']
        train_path_ir = data_dict['train_ir']
        test_path_ir = data_dict['val_ir']
        gs = 32
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
        rank = -1
        
        if opt.image_set == 'train':
            dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, opt.batch_size, gs, opt,
                                                            opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                            world_size=opt.world_size, workers=opt.workers,
                                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                                            dataset_used=opt.dataset_used, temporal_mosaic=opt.temporal_mosaic, 
                                                            use_tadaconv=opt.use_tadaconv, supervision_signal=opt.detection_head, 
                                                            sanitized=opt.sanitized, mosaic=opt.mosaic)
        
        elif opt.image_set == 'val':
            testloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, opt.batch_size * 2, gs, opt,
                                                        opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                        hyp=hyp, cache=opt.cache_images and not opt.notest,rank=-1, #rect=True,# rank=-1,
                                                        world_size=opt.world_size, workers=opt.workers,
                                                        pad=0.5, prefix=colorstr('val: '),
                                                        dataset_used = opt.dataset_used, is_validation=True, 
                                                        use_tadaconv=opt.use_tadaconv, supervision_signal=opt.detection_head,
                                                        sanitized=opt.sanitized) 

        sequences = dataset.res
        img_loc = [sequence[-1] for sequence in sequences ]
        dataset_json = {}
        dataset_json['images'], dataset_json['annotations'],  dataset_json['categories'] = [], [], []

        index = np.arange(0,len(img_loc))
        path_map = dict(zip(img_loc, index))
        bbox_id = 0
        dataset_json['categories'] = [{'id':1, 'name':'person'}]
        for image_path in img_loc:
            dataset_json['images'].append({
                    'id': int(path_map[image_path]),
                    'im_name': image_path,
                    'height': 471,
                    'width': 640
                    }
                        )
            txtfile = image_path.replace('FramesPos','Annotations' ).replace('.tif', '.txt') #saving only the visible images into the json, so that similar to how testing 
            assert 'Visible' in txtfile
            xywh = read_x1_x2_y1_y2_from_txt_annotations(txtfile,name='', use_midxy_wh=False, img_width=640, img_height=471)
            if xywh == []:
                continue

            for ele in xywh:
                x, y, w, h, _ = ele
                dataset_json['annotations'].append({
                        # "type": parts[0],  # Assuming the first part is the type ('person')
                        'id': bbox_id,
                        'image_id': int(path_map[image_path]), 
                        'category_id': 1,
                        'bbox':[ x, y, w, h],
                        'height': h,
                        'occlusion': 0,
                        'ignore': 0,
                        'area':w*h,
                    })
                bbox_id += 1

        json_object = json.dumps(dataset_json, indent=4)

        with open('cvc-14_test_tl_ue.json', 'w') as out:
            out.write(json_object)

