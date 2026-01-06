import re
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Generate train and val dataset for multispectral video object detection')
parser.add_argument('--small', action='store_true', help='Use small dataset')
parser.add_argument('--lframe', type=int, default=3, help='Number of frames in a window')
parser.add_argument('--temporal_stride', type=int, default=3, help='Temporal stride')
parser.add_argument('--midframe', action='store_true')
args = parser.parse_args()

VAL_VIDEO_NAMES =['set00/V002/', 'set00/V006/', 'set01/V001/', 'set02/V002/', 'set04/V001/']
data_path = "/mnt/workspace/datasets/kaist-cvpr15/sanitized_annotations/sanitized_annotations"
lwindow = args.lframe * args.temporal_stride - args.temporal_stride + 1


# root_dir = f"/mnt/workspace/datasets/kaist-cvpr15/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}" + ("_midframe" if args.midframe else "")
root_dir = f"/home/akanu/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}" + ("_midframe" if args.midframe else "")
    
os.makedirs(root_dir, exist_ok=True)

train_dataset = []
val_dataset = []

for filename in tqdm(sorted(os.listdir(data_path))):
    setid, vid, filename = filename.split('_')
    filename = filename.split('.')[0]
    if lwindow > int(filename[1:])+1:
        continue
    ignore = True
    with open(f"{data_path}/{setid}_{vid}_{filename}.txt", "r") as file:
        firstline = file.readline()
        line = file.readline()
        while line:
            line = line.split()
            if line[0] == "person" :
                ignore = False
            line = file.readline()
            
    if not ignore:
        if f"{setid}/{vid}/" in VAL_VIDEO_NAMES:
            val_dataset.append((f"{setid}/{vid}", filename))
        else:
            train_dataset.append((f"{setid}/{vid}", filename))

if args.small:
    val_dataset = val_dataset[:10]
    train_dataset = train_dataset[:50]

def write_to_file(file, dataset, modal):
    global count
    for set_vid, file_name in dataset:
        frame_num = int(file_name[1:])
        if args.midframe:
            start_idx = frame_num - lwindow//2
            end_idx = frame_num + lwindow//2
        else:
            start_idx = frame_num+1-lwindow
            end_idx = frame_num
            
        if start_idx < 0: continue
        if os.path.exists(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/I{end_idx:05d}.jpg"):
            
            file.write(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/{file_name}.jpg\n")


# write the train and val datasets to file
with open(f"{root_dir}/train_vis_kaist_video.txt", "w") as file:
    write_to_file(file, train_dataset, "visible")
        
with open(f"{root_dir}/train_lwir_kaist_video.txt", "w") as file:
    write_to_file(file, train_dataset, "lwir")

with open(f"{root_dir}/val_vis_kaist_video.txt", "w") as file:
    write_to_file(file, val_dataset, "visible")
        
with open(f"{root_dir}/val_lwir_kaist_video.txt", "w") as file:
    write_to_file(file, val_dataset, "lwir")



# Write the train and val datasets to yaml file
yaml_file = f"./data/multispectral_temporal/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}{'_midframe' if args.midframe else ''}.yaml"
with open(yaml_file, "w") as file:
    file.write(f"train_rgb: {root_dir}/train_vis_kaist_video.txt\n")
    file.write(f"val_rgb: {root_dir}/val_vis_kaist_video.txt\n")
    file.write(f"train_ir: {root_dir}/train_lwir_kaist_video.txt\n")
    file.write(f"val_ir: {root_dir}/val_lwir_kaist_video.txt\n")
    file.write("nc: 4\n")
    file.write("names: ['person', 'people', 'cyclist', 'person?']\n")
