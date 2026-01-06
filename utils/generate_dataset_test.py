# Temporary script to generate test dataset for the midframe model. Basically remove the files that do not exist in the dataset for mid frame
import os
from tqdm import tqdm

src = '/home/txt_imageset/test20all_vis.txt'

dst = '/home/txt_imageset/test20all_vis_midframe.txt'

if os.path.exists(dst):
    os.remove(dst)
    
count_all = 0
count = 0

for line in tqdm(open(src)):
    # import pdb; pdb.set_trace()
    # print(line)
    count_all += 1
    line = line.strip()
    idx = int(line.split('/')[-1].split('.')[0][1:])
    file_path = line.replace(f'I{idx:05d}', f'I{idx+3:05d}')
    if os.path.exists(file_path):
        count += 1
        with open(dst, 'a') as f:
            f.write(line + '\n')
            
print("Ratio of valid images: ", count/count_all)
            



src = '/home/txt_imageset/test20all_lwir.txt'

dst = '/home/txt_imageset/test20all_lwir_midframe.txt'

if os.path.exists(dst):
    os.remove(dst)
    
count_all = 0
count = 0

for line in tqdm(open(src)):
    # import pdb; pdb.set_trace()
    # print(line)
    count_all += 1
    line = line.strip()
    idx = int(line.split('/')[-1].split('.')[0][1:])
    file_path = line.replace(f'I{idx:05d}', f'I{idx+3:05d}')
    if os.path.exists(file_path):
        count += 1
        with open(dst, 'a') as f:
            f.write(line + '\n')
            
print("Ratio of valid images: ", count/count_all)
            