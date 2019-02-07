# Overview

This model builds ontop of previous attempts (shout out [dlpbc](https://github.com/dlpbc/comma.ai-speed-challenge) and the
[Stanford team](http://cs229.stanford.edu/proj2017/final-reports/5244226.pdf)) using optical flow and CNN to predict vehicle speed from video.
There are two notable improvements made with my approach:
 1. [Use of a semantic segmentation channel]()
[Dense Optical Flow](https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html) simply tells us the magnitude and direction of motion
in the pixels between two frames.  However this is missing a critical peice of information, which is whether or not that motion is absolute or
relative to the vehicle reference frame.  By adding an additional channel where we label regions that can move relative to the vehicle (ex. other cars, bikes etc.) our model can differentiate and isolate absolute motion of the vehicle in the scene.

 2. [Block based train test split]()
 The next issue is with the train validate split, since adjacent frames in a video tend to share similarities a simple randomized split could
 pollute our validation set by selecting frames similar to those in our training set.  I beleive this is a major issue with all currently
 proposed solutions. The block based splitting approach I chose allows me to scale up the training and validation size while preserving their
 distribution and maintaining a minimum distance between frames. This enables reliable training and validation to be performed on a single video.

## Semantic Segmentation
Semanitic Segmentation is basically pixel level classification of an images. I used the [Berkley Deep Drive Dataset](https://bdd-data.berkeley.edu/) which contained 7k training images where pixels are labeled as one of
[40 classes](https://github.com/NikhilPeri/speedchallenge/blob/master/pipelines/segmentation_preprocess.py#L11-L51). I then relabeled each
image with a binary class representing if given region could move between frames (ex. car, bikes, people) of if it was fixed (ex. tree, road,
bridge) this helped reduce the size of my segmentation model.

![BDD 100k Training Sample](figures/bdd100k_training_sample.png)

I then experimented with two segmentation models, a traditional model based on Transpose Convolutional Layers, and a more complex
[uNet](https://github.com/zhixuhao/unet) which achieved good results on a biomedical imaging dataset. For this dataset I was able to achieve
best results using the *Transpose Convolutional model with 96.5% accuracy and binary cross entropy of 0.109 on a 480x640 image*.

![Comma.ai Training Sample](figures/comma_ai_training_segment.png)

The model had the following architecture
```
todo make diagram of model
```
*Convolutional Layers* was used to downsample the image while increasing the number of channels to preserve information
*Locally Connected* was extremely expensive to implement, it provided great improvement to accuracy since it was able
to learn a filter unique to each region of the image.  Essentially this reduced the probability of detecting the infamous flying car, by
specifying regions in the image where certain classes are expected.
*Transpose Convolutional Layer* this performed "learned upsampling" of the image, it is fairly straight forward and best explained by the
[following lecture](https://youtu.be/nDPWywWRIRo?t=1346).  Essentially we swap the order of our kernel multiplication and take a larger stride
to scale up the image.
*Sigmoid Convolution* is a used to combine all channels with the sigmoid activation function to output a probability for each pixel.
*Bilinear Upsample* this also helps average out single pixel noise since each class must span multiple pixels, however this decreases resolution
our our edges.

We can now construct and input with both optical flow of x and y pixel motion and labeled objects in the scene. This helps reduce the
likelyhood our model will over fit to visual queue in the scene and generally makes it more robust even though the speed model is trained on
a fairly small homogenous dataset.

![Model Input]('figures/hybrid_model_input.png')

This is fed into a simple __ layer CNN which uses Average Pooling for the first __ layers followed by Max Pooling.  While Average Pooling lacks
sensitivity and is generally not used in classifiers, I found the lack of sensitivity in earlier layers was benifitial in this regression model
helping it better generalize to the validation set. In order to demonstrate the effectiveness of the segmentation channel I trained an identical
model using only optical flow and monitored its mean squared error on an identical validation set.

## Block Based Splitter
