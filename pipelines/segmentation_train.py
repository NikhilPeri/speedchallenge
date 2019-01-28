import cv2
import os

def relabel_image(image, classes):
    import pdb; pdb.set_trace()

def resize_image(image):
    import pdb; pdb.set_trace()

def transform_images(input, output):
    images = os.listdir(input)
    for image in images:
        image_data = cv2.imread(os.path.join(input, image))


if __name__ == '__main__':
