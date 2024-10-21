import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model("E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Machine Learning\\CIFER 10\\CIFER10.h5")

plot_model(model,to_file='model.png')
# ANN_VIZ(model, VIEW=TRUE, FILENAME='NETWORK.GV', TITLE='MY NEURAL NETWOR')  