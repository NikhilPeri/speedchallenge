import cv2
import os
import pandas as pd
import numpy as np
from progress.bar import Bar
from multiprocessing import Pool

MAX_CONCURRENCY = 8
bar = Bar('Processing images')
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

def class_is_stationary(color):
    color = tuple(color)
    return stationary_lookup[color]

def processed_label(args):
    label_data = cv2.imread(args[0])
    label_data = cv2.resize(label_data, (0, 0), fx=0.666, fy=0.666, interpolation=cv2.INTER_NEAREST)
    label_data[:, 106:746] # crop 480x852 to 480x640
    label_data = np.apply_along_axis(class_is_stationary, 2, label_datatobia)

    np.save(args[1], label_data)
    bar.next()

def transform_labels(input, output):
    labels = os.listdir(input)
    bar.max = len(labels)

    worker = Pool(MAX_CONCURRENCY)
    worker.map(
        processed_label,
        [(os.path.join(input, label), os.path.join(output, label).replace('_train_color.png', '.npy')) for label in labels]
    )
    bar.finish()

def process_image(args):
    image_data = cv2.imread(args[0])
    image_data = cv2.resize(image_data, (0, 0), fx=0.666, fy=0.666)
    image_data = image_data[:, 106:746] # crop 480x852 to 480x640

    cv2.imwrite(args[1], image_data)
    bar.next()

def transform_images(input, output):
    images = os.listdir(input)
    bar.max = len(images)

    worker = Pool(MAX_CONCURRENCY)
    worker.map(
        process_image,
        [(os.path.join(input, image), os.path.join(output, image)) for image in images]
    )
    bar.finish()

if __name__ == '__main__':
    transform_images('data/bdd100k/segmentation/images/train', 'data/bdd100k/segmentation/processed_images/train')
    transform_images('data/bdd100k/segmentation/images/val', 'data/bdd100k/segmentation/processed_images/val')

    transform_labels('data/bdd100k/segmentation/color_labels/train', 'data/bdd100k/segmentation/processed_labels/train')
    transform_labels('data/bdd100k/segmentation/color_labels/val', 'data/bdd100k/segmentation/processed_labels/val')
