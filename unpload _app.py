# -*- coding: utf-8 -*-
"""unpload.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RYqQ1_uScZSWB7TyEiw8BmbXQDh9c-Xn
"""

from keras.applications.inception_v3 import InceptionV3

import cv2

import cv2
import time
import os

def video_to_frames(input_loc, output_loc):
   # """Function to extract frames from input video file
    #and save them as separate frames in an output directory.
   # Args:
      #  input_loc: Input video file.
      #  output_loc: Output directory to save the frames.
   # Returns:
   #  #   None
    #"""
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":

    input_loc = 'proof.mp4'
    output_loc = 'frames'
    video_to_frames(input_loc, output_loc)

from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
#from google.colab.patches import cv2_imshow

#loading the image to predict
# assign directory
directory = 'frames'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        
      img_path = f
      img = load_img(img_path)
#resize the image to 299x299 square shape
      img = img.resize((299,299))
#convert the image to array
      img_array = img_to_array(img)
#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
      img_array = np.expand_dims(img_array, axis=0)
      #preprocess the input image array
      img_array = preprocess_input(img_array)
#Load the model from internet / computer
#approximately 96 MB
      pretrained_model = InceptionV3(weights="imagenet")
#predict using predict() method
      prediction = pretrained_model.predict(img_array)
#decode the prediction
      actual_prediction = imagenet_utils.decode_predictions(prediction)
      print("predicted object is:")
      print(actual_prediction[0][0][1])
      print("with accuracy")
      print(actual_prediction[0][0][2]*100)
#display image and the prediction text over it
      #cv2.putText(disp_img, actual_prediction[0][0][1], (20,20),
      #cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255,255,255))
#show the image
     # cv2.imshow("Prediction",disp_img)
      #cv2.waitKey(0)

