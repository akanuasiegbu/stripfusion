import json
import os
from os.path import join


# 1 Need to create a folder to save too
# Load the json file and then append to text
def create_save_txtfolder(out_dir, images ):
    """
    timeofday: either 'Day' or 'Night'
    train_or_test: either 'Train'
    """
    os.makedirs(out_dir, exist_ok=True)

    for image in images:
        _image = image['im_name'].split('I')
        folder = _image[0]
        file_name = _image[1]
        
        create_folder_here = join(out_dir, folder)
        os.makedirs( create_folder_here, exist_ok=True)
        create_txt_here = join(create_folder_here, f"I{file_name}.txt")
        with open(create_txt_here,'w') as file: 
            pass
    
    

def write_to_txt(data):
    label = 0
    
    for data_i in data['annotations']:
        if (data_i["occlusion"] == 2) or (data_i["height"] < 55) or data_i["ignore"]:
            continue
        image_id = data_i['image_id']
        
        image = data['images'][image_id]
        _image = image['im_name'].split('I')
        folder = _image[0]
        file_name = _image[1]
        create_folder_here = join(out_dir, folder)
        create_txt_here = join(create_folder_here, f"I{file_name}.txt")
        
        x, y, w,h = data_i['bbox']
        x = x + w/2 
        y = y + h/2
        
        with open(create_txt_here,'a') as file: 
            file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, x/640.,y/512., w/640.,h/512.))
        
        
         
        


if __name__ == '__main__':
    
    out_dir = 'labels_test_from_kaist_annot_reasonable'
    
    with open('./miss_rate_and_map/KAIST_annotation.json', 'r') as file:        
        # Performing this step to ensure reliable comparisons with the KAIST_annotation file,
        # since there are many different annotation formats available for KAIST.
        data = json.load(file) 
    
    create_save_txtfolder(  out_dir=out_dir, images=data['images'] )
    
    write_to_txt(data)