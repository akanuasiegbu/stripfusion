import re
from tqdm import tqdm
import os
import argparse
from os.path import join

"""The annotation format is 
<Frame Number> <Track ID Number > <Annotation Class>
<Bounding box top left x coordinate> <Bounding box top left y coordinate> 
<Bounding Box Width> <Bounding Box Height>"""

def create_save_txtfolder(sequence, out_dir):
    os.makedirs(join(      join(      join(out_dir, 'labels_camel'),    sequence),      'IR'), exist_ok=True)
    os.makedirs(join(      join(      join(out_dir, 'labels_camel'),    sequence),      'Visual'), exist_ok=True)

def traverse_to_folder_and_save_txt_file(images_folder, out_dir=os.getcwd(), norm=True, img_width=336, img_height=256):
    sequences = os.listdir(images_folder)
    sequences.sort()
    for sequence in sequences:
        vid_path = join(images_folder, sequence)
        vid_number = int(sequence[-2:])
        create_save_txtfolder(sequence, out_dir)

        # Read the IR file
        try:
            IR_file_path = join(vid_path, 'Seq{}-IR.txt'.format(vid_number))
            with open(IR_file_path, 'r') as file:
                IR_file = file.readlines()
            # for loop to save IR txt files

            for index, annotation in  enumerate(IR_file):
                frame, track_id, clas, tl_x, tl_y, w, h = annotation.split('\n')[0].split('\t')
                frame, track_id, clas, tl_x, tl_y, w, h = int(frame), int(track_id), int(clas), float(tl_x), float(tl_y), float(w), float(h)
                txt_file = '{:06d}.txt'.format(int(frame))
                with open(join(     join(      join(      join(out_dir, 'labels_camel'),    sequence),      'IR'), txt_file ), 'a') as file:
                    file.write('{} {:.6f} {:.6f} {:.6f} {:.6f} \n'.format(clas, (tl_x+w/2)/img_width, (tl_y+h/2)/img_height, w/img_width, h/img_height))
        except ValueError:
            print('extra space at the end of txt file')
        except FileNotFoundError:
            print('{}IR was not found'.format(sequence))    
        
        # Read the Vis File
        try:
            Vis_file_path = join(vid_path, 'Seq{}-Vis.txt'.format(vid_number))
            with open(Vis_file_path, 'r') as file:
                Vis_file = file.readlines()

            for annotation in Vis_file:
                frame, track_id, clas, tl_x, tl_y, w, h = annotation.split('\n')[0].split('\t')
                frame, track_id, clas, tl_x, tl_y, w, h = int(frame), int(track_id), int(clas), float(tl_x), float(tl_y), float(w), float(h)
                txt_file = '{:06d}.txt'.format(int(frame))
                with open(join(     join(      join(      join(out_dir, 'labels_camel'),    sequence),      'Visual'), txt_file ), 'a') as file:
                    file.write('{} {:.6f} {:.6f} {:.6f} {:.6f} \n'.format(clas, (tl_x+w/2)/img_width, (tl_y+h/2)/img_height, w/img_width, h/img_height))
        except ValueError:
            print('extra space at the end of txt file')
        except FileNotFoundError:
            print("{}Visual was not found".format(sequence))
        


if __name__ == '__main__':
    # know seq 1-5 have txt files
    parser = argparse.ArgumentParser(description='Generate train and val dataset for multispectral video object detection')
    parser.add_argument('--small', action='store_true', help='Use small dataset')
    parser.add_argument('--lframe', type=int, default=3, help='Number of frames in a window')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Temporal stride')
    parser.add_argument('--midframe', action='store_true')
    args = parser.parse_args()

    VAL_VIDEO_NAMES =['sequence_07', 'sequence_09', 'sequence_11', 'sequence_13']
    data_path = '/mnt/workspace/datasets/camel_dataset/images'

    root_dir = f"/home/akanu/camel{'small' if args.small else ''}_lframe{args.lframe}_stride{args.temporal_stride}" + ("_midframe" if args.midframe else "")

    traverse_to_folder_and_save_txt_file(images_folder=data_path)