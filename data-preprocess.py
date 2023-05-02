import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

label = pd.read_csv('IndustryBiscuit/Annotations.csv')

no_defect = label[label['classCode'] == 0 ]['file']
shape_defect = label[label['classCode'] == 1]['file']
object_defect = label[label['classCode'] == 2]['file']
color_defect = label[label['classCode'] == 3]['file']

train_no_defect, test_no_defect = train_test_split(no_defect, test_size=0.1, random_state=42)
train_no_defect, valid_no_defect = train_test_split(train_no_defect, test_size=0.2, random_state=42)

train_shape_defect, test_shape_defect = train_test_split(shape_defect, test_size=0.1, random_state=42)
train_shape_defect, valid_shape_defect = train_test_split(train_shape_defect, test_size=0.2, random_state=42)

train_object_defect, test_object_defect = train_test_split(object_defect, test_size=0.1, random_state=42)
train_object_defect, valid_object_defect = train_test_split(train_object_defect, test_size=0.2, random_state=42)

train_color_defect, test_color_defect = train_test_split(color_defect, test_size=0.1, random_state=42)
train_color_defect, valid_color_defect = train_test_split(train_color_defect, test_size=0.2, random_state=42)

os.makedirs('data/labels/', exist_ok=True)
os.makedirs('data/images/train/no_defect', exist_ok=True)
os.makedirs('data/images/valid/no_defect', exist_ok=True)
os.makedirs('data/images/test/no_defect', exist_ok=True)

os.makedirs('data/images/train/shape_defect', exist_ok=True)
os.makedirs('data/images/valid/shape_defect', exist_ok=True)
os.makedirs('data/images/test/shape_defect', exist_ok=True)

os.makedirs('data/images/train/object_defect', exist_ok=True)
os.makedirs('data/images/valid/object_defect', exist_ok=True)
os.makedirs('data/images/test/object_defect', exist_ok=True)

os.makedirs('data/images/train/color_defect', exist_ok=True)
os.makedirs('data/images/valid/color_defect', exist_ok=True)
os.makedirs('data/images/test/color_defect', exist_ok=True)

def move_image(img_list, dst):
    img_pth = 'IndustryBiscuit/Images/'
    print(f'Moving {len(img_list)} images to {dst}')
    for img in img_list:
        image = os.path.join(img_pth, img)
        os.system(f'mv {image} {dst}')
    print(f'All {len(img_list)} images moved to {dst}')

train_label = pd.DataFrame({'image': np.append(np.append(np.append(train_no_defect, train_shape_defect), train_object_defect), train_color_defect), 
    'label': np.append(np.append(np.append(np.zeros(len(train_no_defect)), np.ones(len(train_shape_defect))), np.ones(len(train_object_defect))*2), np.ones(len(train_color_defect))*3)})  
valid_label = pd.DataFrame({'image': np.append(np.append(np.append(valid_no_defect, valid_shape_defect), valid_object_defect), valid_color_defect), 
    'label': np.append(np.append(np.append(np.zeros(len(valid_no_defect)), np.ones(len(valid_shape_defect))), np.ones(len(valid_object_defect))*2), np.ones(len(valid_color_defect))*3)})  
test_label = pd.DataFrame({'image': np.append(np.append(np.append(test_no_defect, test_shape_defect), test_object_defect), test_color_defect), 
    'label': np.append(np.append(np.append(np.zeros(len(test_no_defect)), np.ones(len(test_shape_defect))), np.ones(len(test_object_defect))*2), np.ones(len(test_color_defect))*3)})  

train_label['label'] = train_label['label'].astype(int)
valid_label['label'] = valid_label['label'].astype(int)
test_label['label'] = test_label['label'].astype(int)

train_label.to_csv('data/labels/train.csv', index=False)
valid_label.to_csv('data/labels/valid.csv', index=False)
test_label.to_csv('data/labels/test.csv', index=False)

move_image(train_no_defect, 'data/images/train/no_defect/')
move_image(train_shape_defect, 'data/images/train/shape_defect/')
move_image(train_object_defect, 'data/images/train/object_defect/')
move_image(train_color_defect, 'data/images/train/color_defect/')

move_image(valid_no_defect, 'data/images/valid/no_defect/')
move_image(valid_shape_defect, 'data/images/valid/shape_defect/')
move_image(valid_object_defect, 'data/images/valid/object_defect/')
move_image(valid_color_defect, 'data/images/valid/color_defect/')

move_image(test_no_defect, 'data/images/test/no_defect/')
move_image(test_shape_defect, 'data/images/test/shape_defect/')
move_image(test_object_defect, 'data/images/test/object_defect/')
move_image(test_color_defect, 'data/images/test/color_defect/')
