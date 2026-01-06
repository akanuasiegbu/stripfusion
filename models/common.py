import yaml
with open('./models/yolov_modules_to_select.yaml') as f:
    select_modules = yaml.safe_load(f)  # data dict

from models.model_modules_shared import *
from models.fusion import *


if select_modules['use_tadaconv']:
    # from models.model_modules_tadaconv import SPP

    from models.model_modules_tadaconv import *
    
    print('Note that the Detect Class is imported to the yolo_test.py directly file')
    print('Using Tadaconv Modules')
else:
    from models.model_modules_concat_cft import *
    # from models.model_modules_fusion import GPT
    print('Note that the Detect Class is imported to the yolo_test.py directly file')
    print('Using Concatnated CFT modules')