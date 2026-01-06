import json
import os
from os.path import join


# 1 Need to create a folder to save too
# Load the json file and then append to text
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
    # os.makedirs( join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday,'FIR', train_or_test)), exist_ok=True)
    
    

def write_to_txt(data, scale_to_640_512):
    label = 0
    for data_i in data['images']:
        
        im_name = data_i['im_name'].split('/')
        
        timeofday = 'Night' if 'Night' in im_name[5] else 'Day'
        _timeofday = timeofday if not scale_to_640_512 else f'{timeofday}_Resized'
        
        vis_or_ir = 'Visible' if 'Visible' in im_name[6] else 'FIR'
        
        
        txt_name = im_name[-1].split('.')[0] + '.txt'
        with open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday, vis_or_ir, train_or_test)), txt_name),'w') as file: 
            pass
    
    
    for data_i in data['annotations']:
        if data_i['height'] < 55:
            continue
        image_id = data_i['image_id']
        
        im_name = data['images'][image_id]['im_name'].split('/')
        
        timeofday = 'Night' if 'Night' in im_name[5] else 'Day'
        _timeofday = timeofday if not scale_to_640_512 else f'{timeofday}_Resized'
        
        vis_or_ir = 'Visible' if 'Visible' in im_name[6] else 'FIR'
        
        x, y, w,h = data_i['bbox']
        x = x + w/2 if not scale_to_640_512 else (x + w/2)/640
        y = y + h/2 if not scale_to_640_512 else (1.087044832*(y + h/2))/512
        
        w = w if not scale_to_640_512 else w/640
        h = h if not scale_to_640_512 else (1.087044832*h)/512
        
        txt_name = im_name[-1].split('.')[0] + '.txt'
        with open(join(join(out_dir, 'labels_cvc14/{}/{}/{}/FramesPos'.format(_timeofday, vis_or_ir, train_or_test)), txt_name),'a') as file: 
            file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, x,y, w,h))  


if __name__ == '__main__':
    
    out_dir = 'datatosave'
    train_or_test = 'NewTest'
    scale_to_640_512 = True
    
    with open('./json_gt/cvc-14_test_tl.json', 'r') as file:
        data = json.load(file)
        
    
    create_save_txtfolder('Day', 
                          out_dir=out_dir, 
                          train_or_test=train_or_test, 
                          scale_to_640_512=scale_to_640_512)
    
    create_save_txtfolder('Night', 
                        out_dir=out_dir, 
                        train_or_test=train_or_test, 
                        scale_to_640_512=scale_to_640_512)
    
    write_to_txt(data, scale_to_640_512)