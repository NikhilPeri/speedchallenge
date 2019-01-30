import cv2
import os
import pandas as pd
import numpy as np
from progress.bar import Bar

classes = pd.DataFrame([
    ['unlabled'          ,  'void'        , (  0,  0,  0),  0],
    ['dynamic'           ,  'void'        , (  0, 74,111),  1],
    ['ego vehicle'       ,  'void'        , (  0,  0,  0),  0],
    ['ground'            ,  'void'        , ( 81,  0, 81),  1],
    ['static'            ,  'void'        , (  0,  0,  0),  1],
    ['parking'           ,  'flat'        , (160,170,250),  1],
    ['rail track'        ,  'flat'        , (140,150,230),  1],
    ['road'              ,  'flat'        , (128, 64,128),  1],
    ['sidewalk'          ,  'flat'        , (232, 35,244),  1],
    ['bridge'            ,  'construction', (100,100,150),  1],
    ['building'          ,  'construction', ( 70, 70, 70),  1],
    ['fence'             ,  'construction', (153,153,190),  1],
    ['garage'            ,  'construction', (180,100,180),  1],
    ['guard rail'        ,  'construction', (180,165,180),  1],
    ['tunnel'            ,  'construction', ( 90,120,150),  1],
    ['wall'              ,  'construction', (156,102,102),  1],
    ['banner'            ,  'object'      , (100,170,250),  1],
    ['billboard'         ,  'object'      , (250,220,220),  1],
    ['lane divider'      ,  'object'      , ( 0, 165,255),  1],
    ['parking sign'      ,  'object'      , ( 60, 20,220),  1],
    ['pole'              ,  'object'      , (153,153,153),  1],
    ['polegroup'         ,  'object'      , (153,153,153),  1],
    ['street light'      ,  'object'      , (100,220,220),  1],
    ['traffic cone'      ,  'object'      , (  0, 70,255),  1],
    ['traffic device'    ,  'object'      , (220,220,220),  1],
    ['traffic light'     ,  'object'      , ( 30,170,250),  1],
    ['traffic sign'      ,  'object'      , (  0,220,220),  1],
    ['traffic sign frame',  'object'      , (250,170,250),  1],
    ['terrain'           ,  'nature'      , (152,251,152),  1],
    ['vegetation'        ,  'nature'      , ( 35,142,107),  1],
    ['sky'               ,  'sky'         , (180,130, 70),  1],
    ['person'            ,  'human'       , ( 60, 20,220),  0],
    ['rider'             ,  'human'       , (  0,  0,255),  0],
    ['bicycle'           ,  'vehicle'     , ( 32, 11,119),  0],
    ['bus'               ,  'vehicle'     , (100, 60,  0),  0],
    ['car'               ,  'vehicle'     , (142,  0,  0),  0],
    ['caravan'           ,  'vehicle'     , ( 90,  0,  0),  0],
    ['motorcycle'        ,  'vehicle'     , (230,  0,  0),  0],
    ['trailer'           ,  'vehicle'     , (110,  0,  0),  0],
    ['train'             ,  'vehicle'     , (100, 80,  0),  0],
    ['truck'             ,  'vehicle'     , ( 70,  0,  0),  0]],
    columns=['name', 'category', 'color', 'isStationary'],
)
stationary_lookup = pd.Series(classes['isStationary'].values, classes['color']).to_dict()

def resize_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (0, 0), fx=0.666, fy=0.666)
    return image[:, 106:746] # crop 480x852 to 480x640

def resize_label(label):
    label = cv2.resize(label, (0, 0), fx=0.666, fy=0.666, interpolation=cv2.INTER_NEAREST)
    return label[:, 106:746] # crop 480x852 to 480x640

def class_is_stationary(color):
    color = tuple(color)
    return stationary_lookup[color]

def reclassify_label(label):
    label = cv2.resize(label, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    label = np.apply_along_axis(class_is_stationary, 2, label)
    label = cv2.resize(label, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    return label

def transform_images(input, output):
    images = os.listdir(input)
    bar = Bar('Processing Labels', max=len(labels))
    for image in images:
        image_data = cv2.imread(os.path.join(input, image))
        image_data = resize_image(image_data)
        cv2.imwrite(os.path.join(output, image), image_data)
        print "Processed {}".format(image)
        bar.next()
    bar.finish()

def transform_labels(classes, input, output):
    labels = os.listdir(input)
    bar = Bar('Processing Labels', max=len(labels))
    for label in labels:
        label_data = cv2.imread(os.path.join(input, label))
        label_data = resize_label(label_data)
        label_data = reclassify_label(label_data)
        np.save(os.path.join(output, label.replace('_color.png', '.npy')), label_data)
        bar.next()
    bar.finish()

if __name__ == '__main__':
    transform_images('data/bdd100k/segmentation/images/train', 'data/bdd100k/stationary/images/train')
    transform_images('data/bdd100k/segmentation/images/val', 'data/bdd100k/stationary/images/val')

    #transform_labels(classes, 'data/bdd100k/segmentation/color_labels/train', 'data/bdd100k/stationary/labels/train')
    #transform_labels(classes, 'data/bdd100k/segmentation/color_labels/val', 'data/bdd100k/stationary/labels/val')
