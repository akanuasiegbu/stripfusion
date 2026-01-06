#Obtain testing data from https://github.com/CalayZhou/MBNet/issues/28
# https://github.com/CalayZhou/MBNet/files/5352777/CVC14_txt_to_numpy.txt

#coding=utf-8
import numpy as np
import os
import json
#kaist_example = np.load("22158train_data_all4.npy")
#print("done!!")
###################
#训练集标签制作
###################
image_train_num_txthave_read = 0
image_train_num_none = 0 
image_train_lens_notequal_none = 0
image_train_num=0
image_train_person_num = 0 
image_train_list=[]

pedestrian_aligned_num = 0 
pedestrian_notaligned_num = 0  

path_CVC14_train_day_visible = "/mnt/workspace/datasets/CVC-14/Day/Visible/NewTest/Annotations"
path_CVC14_train_day_FIR = "/mnt/workspace/datasets/CVC-14/Day/FIR/NewTest/Annotations"
path_CVC14_train_night_visible = "/mnt/workspace/datasets/CVC-14/Night/Visible/NewTest/Annotations"
path_CVC14_train_night_FIR = "/mnt/workspace/datasets/CVC-14/Night/FIR/NewTest/Annotations"
# image_path="data/CVC-14"
image_path = '/mnt/workspace/datasets/CVC-14'
txt_file_path_visible = []
txt_file_path_fir = []

global scale_to_640_512
scale_to_640_512 = True

def read_x1_x2_y1_y2_from_txt_annotations(path,name='', scale_to_640_512=False):
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
                bbox_info_one = [0, 0, 0, 0,0]
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
                bbox_info_one[0] = x1 #top left
                bbox_info_one[1] = y1 if not scale_to_640_512 else y1*1.087044832 #top left
                bbox_info_one[2] = w1
                bbox_info_one[3] = h1 if not scale_to_640_512 else h1*1.087044832
                bbox_info_one[4] = 0 # 0 means only using person
                bbox_info.append(bbox_info_one)
    return bbox_info
#bbox_info_night_visible与bbox_info_night_FIR长度不一致
def bbox_info_combine_aligned(bbox_info_night_visible,bbox_info_night_FIR,samplename=''):
    global pedestrian_aligned_num  # 对齐行人的数量
    global pedestrian_notaligned_num  # 没对齐行人的数量

    if len(bbox_info_night_visible)==len(bbox_info_night_FIR):
        assert len(bbox_info_night_visible)==len(bbox_info_night_FIR)#断言长度是否一致
        pedestrian_num = len(bbox_info_night_visible)
        bbox_info_array = np.zeros((pedestrian_num, 4))
        for bbox_i in range(pedestrian_num):
            x1 = bbox_info_night_visible[bbox_i][0]
            y1 =  int(float(bbox_info_night_visible[bbox_i][1]) ) #* 1.087044832) #bbox_info_night_visible[bbox_i][1]
            x2 = bbox_info_night_visible[bbox_i][2]
            y2 = int(float(bbox_info_night_visible[bbox_i][3]) ) #* 1.087044832)#bbox_info_night_visible[bbox_i][3]
            pedestrian_aligned_num=pedestrian_aligned_num+1
            bbox_info_array[bbox_i, 0] = x1
            bbox_info_array[bbox_i, 1] = y1
            bbox_info_array[bbox_i, 2] = x2
            bbox_info_array[bbox_i, 3] = y2
            # x1_fir = bbox_info_night_FIR[bbox_i][0]
            # y1_fir = bbox_info_night_FIR[bbox_i][1]
            # x2_fir = bbox_info_night_FIR[bbox_i][2]
            # y2_fir = bbox_info_night_FIR[bbox_i][3]
            #
            # #计算iou
            # x_min_out = min(x1,x1_fir)
            # x_min_in = max(x1,x1_fir)
            # y_min_out = min(y1,y1_fir)
            # y_min_in = max(y1,y1_fir)
            #
            # x_max_out = max(x2,x2_fir)
            # x_max_in = min(x2,x2_fir)
            # y_max_out = max(y2,y2_fir)
            # y_max_in = min(y2,y2_fir)
            #
            # Iou_Large = (x_max_out - x_min_out)*(y_max_out - y_min_out)
            # Iou_Small = (x_max_in - x_min_in)*(y_max_in - y_min_in)
            # Iou = float(Iou_Small)/float(Iou_Large)
            # if(Iou<0.5):
            #     #print(samplename+" FIR and RGB not aligned very vell,the IoU is ", Iou)
            #     pedestrian_notaligned_num = pedestrian_notaligned_num + 1
            #     bbox_info_array[bbox_i, 0] = (x1 + x1_fir) // 2
            #     bbox_info_array[bbox_i, 1] = int(float((y1 + y1_fir) // 2)*1.087044832)#512/471
            #     bbox_info_array[bbox_i, 2] = (x2 + x2_fir) // 2
            #     bbox_info_array[bbox_i, 3] = int(float((y2 + y2_fir) // 2)*1.087044832)
            # else:
            #     pedestrian_aligned_num=pedestrian_aligned_num+1
            #     bbox_info_array[bbox_i, 0] = (x1 + x1_fir) // 2
            #     bbox_info_array[bbox_i, 1] = int(float((y1 + y1_fir) // 2)*1.087044832)#512/471
            #     bbox_info_array[bbox_i, 2] = (x2 + x2_fir) // 2
            #     bbox_info_array[bbox_i, 3] = int(float((y2 + y2_fir) // 2)*1.087044832)#512/471

    elif len(bbox_info_night_visible)!=len(bbox_info_night_FIR):
        pedestrian_num = len(bbox_info_night_visible)
        bbox_info_array = np.zeros((pedestrian_num, 4))
        for bbox_i in range(len(bbox_info_night_visible)):
            pedestrian_aligned_num = pedestrian_aligned_num + 1
            bbox_info_array[bbox_i, 0] = bbox_info_night_visible[bbox_i][0]
            bbox_info_array[bbox_i, 1] = int(float(bbox_info_night_visible[bbox_i][1]) ) # * 1.087044832)
            bbox_info_array[bbox_i, 2] = bbox_info_night_visible[bbox_i][2]
            bbox_info_array[bbox_i, 3] = int(float(bbox_info_night_visible[bbox_i][3]) ) # * 1.087044832)

        # if(len(bbox_info_night_visible)>len(bbox_info_night_FIR)):
        #     bbox_info_large=bbox_info_night_visible
        #     bbox_info_small=bbox_info_night_FIR
        #     pedestrian_num=len(bbox_info_night_visible)
        # if(len(bbox_info_night_visible)<len(bbox_info_night_FIR)):
        #     bbox_info_small=bbox_info_night_visible
        #     bbox_info_large=bbox_info_night_FIR
        #     pedestrian_num = len(bbox_info_night_FIR)
        # bbox_info_array = np.zeros((pedestrian_num, 4))
        #
        # if(len(bbox_info_small)==0):# 一个有标注，一个没有标注的情况
        #     for bbox_i in range(pedestrian_num):
        #         pedestrian_aligned_num = pedestrian_aligned_num + 1
        #         bbox_info_array[bbox_i, 0] = bbox_info_large[bbox_i][0]
        #         bbox_info_array[bbox_i, 1] = int(float(bbox_info_large[bbox_i][1]) * 1.087044832)
        #         bbox_info_array[bbox_i, 2] = bbox_info_large[bbox_i][2]
        #         bbox_info_array[bbox_i, 3] = int(float(bbox_info_large[bbox_i][3]) * 1.087044832)
        # else:
        #     for bbox_i in range(pedestrian_num):
        #         for bbox_j in range(len(bbox_info_small)):#编号一致相加除2
        #             if(bbox_info_large[bbox_i][4] == bbox_info_small[bbox_j][4] ):
        #                 x1 = bbox_info_large[bbox_i][0]
        #                 y1 = bbox_info_large[bbox_i][1]
        #                 x2 = bbox_info_large[bbox_i][2]
        #                 y2 = bbox_info_large[bbox_i][3]
        #
        #                 x1_fir = bbox_info_small[bbox_j][0]
        #                 y1_fir = bbox_info_small[bbox_j][1]
        #                 x2_fir = bbox_info_small[bbox_j][2]
        #                 y2_fir = bbox_info_small[bbox_j][3]
        #
        #                 # 计算iou
        #                 x_min_out = min(x1, x1_fir)
        #                 x_min_in = max(x1, x1_fir)
        #                 y_min_out = min(y1, y1_fir)
        #                 y_min_in = max(y1, y1_fir)
        #
        #                 x_max_out = max(x2, x2_fir)
        #                 x_max_in = min(x2, x2_fir)
        #                 y_max_out = max(y2, y2_fir)
        #                 y_max_in = min(y2, y2_fir)
        #
        #                 Iou_Large = (x_max_out - x_min_out) * (y_max_out - y_min_out)
        #                 Iou_Small = (x_max_in - x_min_in) * (y_max_in - y_min_in)
        #                 Iou = float(Iou_Small) / float(Iou_Large)
        #                 if (Iou < 0.5):
        #                     # print(samplename+" FIR and RGB not aligned very vell,the IoU is ", Iou)
        #                     pedestrian_notaligned_num = pedestrian_notaligned_num + 1
        #                     bbox_info_array[bbox_i, 0] = (x1 + x1_fir) // 2
        #                     bbox_info_array[bbox_i, 1] = int(float((y1 + y1_fir) // 2)*1.087044832)
        #                     bbox_info_array[bbox_i, 2] = (x2 + x2_fir) // 2
        #                     bbox_info_array[bbox_i, 3] = int(float((y2 + y2_fir) // 2)*1.087044832)
        #                 else:
        #                     pedestrian_aligned_num = pedestrian_aligned_num + 1
        #                     bbox_info_array[bbox_i, 0] = (x1 + x1_fir) // 2
        #                     bbox_info_array[bbox_i, 1] = int(float((y1 + y1_fir) // 2)*1.087044832)
        #                     bbox_info_array[bbox_i, 2] = (x2 + x2_fir) // 2
        #                     bbox_info_array[bbox_i, 3] = int(float((y2 + y2_fir) // 2)*1.087044832)
        #             else:#编号不一致取有编号的一项
        #                 pedestrian_aligned_num = pedestrian_aligned_num + 1
        #                 bbox_info_array[bbox_i, 0] = bbox_info_large[bbox_i][0]
        #                 bbox_info_array[bbox_i, 1] = int(float(bbox_info_large[bbox_i][1])*1.087044832)
        #                 bbox_info_array[bbox_i, 2] = bbox_info_large[bbox_i][2]
        #                 bbox_info_array[bbox_i, 3] = int(float(bbox_info_large[bbox_i][3])*1.087044832)

    return bbox_info_array






#白天
# path_CVC14_train_day_visible.sort()
for train_day_visible_i in os.listdir(path_CVC14_train_day_visible)  :#set
    image_train_num_txthave_read=image_train_num_txthave_read+1
    path_day_visible_sample = path_CVC14_train_day_visible + '/' + train_day_visible_i
    path_day_FIR_sample = path_CVC14_train_day_FIR + '/' + train_day_visible_i


    image_path = path_day_visible_sample.replace('Annotations', "FramesPos" )
    image_path = image_path.replace('.txt', '.tif')
    image_path_fir = image_path.replace('Visible', 'FIR')
                                            
    if not os.path.exists(path_day_FIR_sample) or not os.path.exists(image_path) or not os.path.exists(image_path_fir):#红外不存在标签
        "If the annotation of fir or the visible image path or the fir image path does not exist do not read that annotation"
        print(f"visible but no fir path: {path_day_FIR_sample}")
        # print("--------------don't exsit in the day FIR path-------------------")
        continue
    else:
        txt_file_path_visible.append(path_day_visible_sample)
        txt_file_path_fir.append(path_day_FIR_sample)
        
    bbox_info_day_visible = read_x1_x2_y1_y2_from_txt_annotations(path_day_visible_sample, name='day_visible')
    bbox_info_day_FIR = read_x1_x2_y1_y2_from_txt_annotations(path_day_FIR_sample, name='day_FIR')

    if(len(bbox_info_day_visible)==0 and len(bbox_info_day_FIR)==0):
        #通过判断文件大小是否为0
        bbox_info_day_visible_size = os.path.getsize(path_day_visible_sample)
        bbox_info_day_FIR_size = os.path.getsize(path_day_FIR_sample)
        if ( bbox_info_day_visible_size==0 and bbox_info_day_FIR_size==0):
            image_train_num_none=image_train_num_none+1
            image_train_list_one = {'bboxes': [],
                                    'filepath_lwir': image_path + '/Day/FIR/NewTest/FramesPos/' + train_day_visible_i[
                                                                                                :-4] + '.tif',
                                    'filepath': image_path + '/Day/Visible/NewTest/FramesPos/' + train_day_visible_i[
                                                                                               :-4] + '.tif',
                                    'filepath_large_seg': '',
                                    'filepath_small_seg': '',
                                    'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
            image_train_list.append(image_train_list_one)

        continue
        print("--------------" + path_day_visible_sample + " is null--------------")

    if (len(bbox_info_day_visible)  != len(bbox_info_day_FIR) ):
        image_train_lens_notequal_none = image_train_lens_notequal_none+1
        bbox_info_array = bbox_info_combine_aligned(bbox_info_day_visible,bbox_info_day_FIR,samplename=train_day_visible_i)
        image_train_list_one = {'bboxes': bbox_info_array,
                                'filepath_lwir': image_path + '/Day/FIR/NewTest/FramesPos/' +train_day_visible_i[:-4] + '.tif',
                                'filepath': image_path + '/Day/Visible/NewTest/FramesPos/' +train_day_visible_i[:-4] + '.tif',
                                'filepath_large_seg': '',
                                'filepath_small_seg': '',
                                'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
        image_train_list.append(image_train_list_one)
        #print("--------------" + path_day_visible_sample + " len not equal--------------")

    if (len(bbox_info_day_visible) == len(bbox_info_day_FIR) ):
        image_train_num = image_train_num + 1
        bbox_info_array = bbox_info_combine_aligned(bbox_info_day_visible,bbox_info_day_FIR,samplename=train_day_visible_i)
        image_train_list_one = {'bboxes': bbox_info_array,
                                'filepath_lwir': image_path + '/Day/FIR/NewTest/FramesPos/' +train_day_visible_i[:-4] + '.tif',
                                'filepath': image_path + '/Day/Visible/NewTest/FramesPos/' +train_day_visible_i[:-4] + '.tif',
                                'filepath_large_seg': '',
                                'filepath_small_seg': '',
                                'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
        image_train_list.append(image_train_list_one)

# print("all image_train_num_txthave_read",image_train_num_txthave_read)
# print("image_train_num_none",image_train_num_none)
# print("image_train_lens_notequal_none",image_train_lens_notequal_none)
# print("extract done!,the total train num is ",image_train_num)

# print("------------------------pedestrian num----------------------------------------")
# print("pedestrian_notaligned_num ",pedestrian_notaligned_num)
# print("pedestrian_aligned_num ",pedestrian_aligned_num)
# print("------------------------pedestrian num----------------------------------------")
#夜晚
# path_CVC14_train_night_visible.sort()
for train_night_visible_i in os.listdir(path_CVC14_train_night_visible):  # set
    image_train_num_txthave_read = image_train_num_txthave_read + 1
    path_night_visible_sample = path_CVC14_train_night_visible + '/' + train_night_visible_i
    path_night_FIR_sample = path_CVC14_train_night_FIR + '/' + train_night_visible_i
    
    image_path = path_night_visible_sample.replace('Annotations', "FramesPos" )
    image_path = image_path.replace('.txt', '.tif')
    image_path_fir = image_path.replace('Visible', 'FIR')

    if not os.path.exists(path_night_FIR_sample) or not os.path.exists(image_path) or not os.path.exists(image_path_fir):  # 红外不存在标签
        print(f"visible but not fir path {path_night_FIR_sample}")
        # print("--------------don't exsit in the day FIR path-------------------")
        continue
    else:
        txt_file_path_visible.append(path_night_visible_sample)
        txt_file_path_fir.append(path_night_FIR_sample)
        
    bbox_info_night_visible = read_x1_x2_y1_y2_from_txt_annotations(path_night_visible_sample, name='night_visible')
    bbox_info_night_FIR = read_x1_x2_y1_y2_from_txt_annotations(path_night_FIR_sample, name='night_FIR')

    if (len(bbox_info_night_visible) == 0 and len(bbox_info_night_FIR) == 0):

        bbox_info_night_visible_size = os.path.getsize(path_night_visible_sample)
        bbox_info_night_FIR_size = os.path.getsize(path_night_FIR_sample)
        if ( bbox_info_night_visible_size==0 and bbox_info_night_FIR_size==0):
            image_train_num_none=image_train_num_none+1
            image_train_list_one = {'bboxes': [],
                                    'filepath_lwir': image_path + '/Night/FIR/NewTest/FramesPos/' + train_night_visible_i[
                                                                                                :-4] + '.tif',
                                    'filepath': image_path + '/Night/Visible/NewTest/FramesPos/' + train_night_visible_i[
                                                                                               :-4] + '.tif',
                                    'filepath_large_seg': '',
                                    'filepath_small_seg': '',
                                    'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
            image_train_list.append(image_train_list_one)
        continue
        print("--------------" + path_night_visible_sample + " is null--------------")

    if (len(bbox_info_night_visible) != len(bbox_info_night_FIR) ):
        #image_train_num = image_train_num + 1
        image_train_lens_notequal_none = image_train_lens_notequal_none + 1
        bbox_info_array = bbox_info_combine_aligned(bbox_info_night_visible,bbox_info_night_FIR,samplename=train_night_visible_i)
        image_train_list_one = {'bboxes': bbox_info_array,
                                'filepath_lwir': image_path + '/Night/FIR/NewTest/FramesPos/' +train_night_visible_i[:-4] + '.tif',
                                'filepath': image_path + '/Night/Visible/NewTest/FramesPos/' +train_night_visible_i[:-4] + '.tif',
                                'filepath_large_seg': '',
                                'filepath_small_seg': '',
                                'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
        image_train_list.append(image_train_list_one)
        #print("--------------" + path_night_visible_sample + " len not equal--------------")

    if( len(bbox_info_night_visible) == len(bbox_info_night_FIR)):
        image_train_num = image_train_num + 1
        bbox_info_array = bbox_info_combine_aligned(bbox_info_night_visible,bbox_info_night_FIR,samplename=train_night_visible_i)
        image_train_list_one = {'bboxes': bbox_info_array,
                                'filepath_lwir': image_path + '/Night/FIR/NewTest/FramesPos/' +train_night_visible_i[:-4] + '.tif',
                                'filepath': image_path + '/Night/Visible/NewTest/FramesPos/' +train_night_visible_i[:-4] + '.tif',
                                'filepath_large_seg': '',
                                'filepath_small_seg': '',
                                'ignoreareas': np.array([]), 'vis_bboxes': np.array([])}
        image_train_list.append(image_train_list_one)



print("all image_train_num_txthave_read",image_train_num_txthave_read)
print("image_train_num_none",image_train_num_none)
print("image_train_lens_notequal_none",image_train_lens_notequal_none)
print("extract done!,the total train num is ",image_train_num)
print("------------------------pedestrian num----------------------------------------")
print("pedestrian_notaligned_num ",pedestrian_notaligned_num)
print("pedestrian_aligned_num ",pedestrian_aligned_num)
print("extract done!,the total train pedestrian is ",pedestrian_aligned_num+pedestrian_notaligned_num)
print("------------------------pedestrian num----------------------------------------")
image_train_list_to_array= np.array(image_train_list)
#np.save('CVC14_TestList.npy', image_train_list_to_array)
print("done")



# This creates the txt file for the Yaml file
txt_file_path_visible.sort()
txt_file_path_fir.sort()
with open('cvc14_test_visible.txt', 'w') as file:
    for ann_path in txt_file_path_visible:
    # fir_name = image_path.split('/')[-1].split('.')[0]
    # if fir_name not in visible_set:
        image_path = ann_path.replace('Annotations', "FramesPos" )
        image_path = image_path.replace('.txt', '.tif')
        image_path_fir = image_path.replace('Visible', 'FIR')
        
        if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
            image_path = image_path.replace('/Day/', '/Day_Resized/')
        elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
            image_path = image_path.replace('/Night/', '/Night_Resized/')

        if os.path.exists(image_path) and os.path.exists(image_path_fir):
            file.write('{}\n'.format(image_path))
        
            
with open('cvc14_test_fir.txt', 'w') as file:
    for ann_path in txt_file_path_fir:
    # visible_name = image_path.split('/')[-1].split('.')[0]
    # if visible_name not in fir_set:
        image_path_fir = ann_path.replace('Annotations', "FramesPos" )
        image_path_fir = image_path_fir.replace('.txt', '.tif')
        image_path = image_path_fir.replace('FIR', 'Visible')
        if scale_to_640_512 and 'Day' in image_path_fir.split('/')[5]:
            image_path_fir = image_path_fir.replace('/Day/', '/Day_Resized/')
        elif scale_to_640_512 and 'Night' in image_path_fir.split('/')[5]:
            image_path_fir = image_path_fir.replace('/Night/', '/Night_Resized/')

        if os.path.exists(image_path) and os.path.exists(image_path_fir):
            file.write('{}\n'.format(image_path_fir))

def to_json_for_eval(visible_path, fir_path, image_id = 0, bbox_id = 0,
                      txt_file_path_visible=[], txt_file_path_fir=[], images=[], annotations=[], scale_to_640_512=False):
    # path_CVC14_visible.sort()
    vis_dir_list = os.listdir(visible_path)
    vis_dir_list.sort()
    for train_day_visible_i in vis_dir_list:#set
        # image_train_num_txthave_read=image_train_num_txthave_read+1
        path_day_visible_sample = visible_path + '/' + train_day_visible_i
        path_day_FIR_sample = fir_path + '/' + train_day_visible_i


        image_path = path_day_visible_sample.replace('Annotations', "FramesPos" )
        image_path = image_path.replace('.txt', '.tif')
        image_path_fir = image_path.replace('Visible', 'FIR')
                                                
        if not os.path.exists(path_day_FIR_sample) or not os.path.exists(image_path) or not os.path.exists(image_path_fir):#红外不存在标签
            "If the annotation of fir or the visible image path or the fir image path does not exist do not read that annotation"
            print(f"visible but no fir path: {path_day_FIR_sample}")
            # print("--------------don't exsit in the day FIR path-------------------")
            continue
        else:
            txt_file_path_visible.append(path_day_visible_sample)
            txt_file_path_fir.append(path_day_FIR_sample)
            
        bbox_info_day_visible = read_x1_x2_y1_y2_from_txt_annotations(path_day_visible_sample, name='day_visible', scale_to_640_512=scale_to_640_512)
        bbox_info_day_FIR = read_x1_x2_y1_y2_from_txt_annotations(path_day_FIR_sample, name='day_FIR', scale_to_640_512=scale_to_640_512)
        
        if scale_to_640_512 and 'Day' in image_path.split('/')[5]:
            image_path = image_path.replace('/Day/', '/Day_Resized/')
        elif scale_to_640_512 and 'Night' in image_path.split('/')[5]:
            image_path = image_path.replace('/Night/', '/Night_Resized/')

        images.append({
                        'id': int(image_id),
                        'im_name': image_path,
                        'height': 471 if not scale_to_640_512 else 512,
                        'width': 640
                        }
                            )

            
        if(len(bbox_info_day_visible)==0 and len(bbox_info_day_FIR)==0):
            pass
        else:
            bbox_info_array = bbox_info_combine_aligned(bbox_info_day_visible,bbox_info_day_FIR,samplename=train_day_visible_i)
            for bbox in bbox_info_array:
                x,y,w,h, = bbox

                image_train_list_one = {    'id': bbox_id,
                                            'image_id': int(image_id), 
                                            'category_id': 1,
                                            'bbox':[ x, y, w, h],
                                            'height': h,
                                            'occlusion': 0, # need to double check this
                                            'ignore': 0,
                                            'area':w*h,
                                            }
                annotations.append(image_train_list_one)
                bbox_id +=1
        
        image_id += 1
    
    return images, annotations, image_id , bbox_id, txt_file_path_visible, txt_file_path_fir

if __name__ == '__main__':
    
    print(f'scale to 640 x 512 {scale_to_640_512}')
    images, ann, image_id , bbox_id, txt_file_path_visible, txt_file_path_fir = to_json_for_eval(
        visible_path =  "/mnt/workspace/datasets/CVC-14/Day/Visible/NewTest/Annotations",
        fir_path = "/mnt/workspace/datasets/CVC-14/Day/FIR/NewTest/Annotations",
        scale_to_640_512 = scale_to_640_512
    )

    images, ann, image_id , bbox_id, txt_file_path_visible, txt_file_path_fir = to_json_for_eval(visible_path="/mnt/workspace/datasets/CVC-14/Night/Visible/NewTest/Annotations",
                                                                                                 fir_path="/mnt/workspace/datasets/CVC-14/Night/FIR/NewTest/Annotations",
                                                                                                 image_id=image_id,
                                                                                                 bbox_id=bbox_id,
                                                                                                 txt_file_path_visible=txt_file_path_visible,
                                                                                                 txt_file_path_fir=txt_file_path_fir,
                                                                                                 images=images,
                                                                                                 annotations=ann,
                                                                                                 scale_to_640_512 = scale_to_640_512
                                                                                                 )
    
    val_path_rgb = '/mnt/workspace/datasets/CVC-14/cvc14_test_visible.txt'
    file = open(val_path_rgb) # visible path
    paths = file.read().splitlines()

    file_new = open(val_path_rgb)
    paths_new = file_new.read().splitlines()
    # paths.sort()
    index = np.arange(0,1417)
    path_map = dict(zip(paths, index))

    dataset = {}
    dataset['images'] = images
    dataset['annotations'] = ann
    dataset['categories'] = [{'id':1, 'name':'person'}]

    json_object = json.dumps(dataset, indent=4)


    with open('cvc-14_test_tl_scale_height.json', 'w') as out:
        out.write(json_object)
    print('here')