import cv2
import dask.array as da
import numpy as np
from keras.models import load_model
from progress.bar import Bar


segmentation_model = load_model('models/segmentation.hdf5')
def segment_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.reshape(480, 640, 1)
    frame = segmentation_model.predict(np.array([frame]))[0]
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
    frame = cv2.blur(frame, (3, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame

def compute_frame_segments(input, output):
    input = cv2.VideoCapture(input)
    bar = Bar("Processed Frame", max=input.get(cv2.CAP_PROP_FRAME_COUNT))
    import pdb; pdb.set_trace()
    while input.isOpened():
        valid, frame = input.read()
        if not valid:
            break
        cv2.imwrite(os.path.join(output, '{}.jpg'.format(bar.index), segment_frame(frame)))
        bar.next()

    output.release()

if __name__ == '__main__':
    compute_frame_segments('data/comma_ai/train.mp4', 'data/comma_ai/train_segments')
    compute_frame_segments('data/comma_ai/test.mp4', 'data/comma_ai/test_segments')
