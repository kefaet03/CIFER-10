import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Machine Learning\\CIFER 10\\CIFER10.h5")

labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    
    return predicted_label

def output_pic(image,label):
    cv2.putText(image,label,(5,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image

image_path = "E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Machine Learning\\CIFER 10\\Picture\\05.png"
img = cv2.imread(image_path)

predicted_label = predict_image(image_path)
print(f"The predicted label is: {predicted_label}")

img_copy = img.copy()
output_img = output_pic(img_copy,predicted_label)


cv2.imshow("Input Image", img)
cv2.imshow("Ouput image",output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# "E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Machine Learning\\CIFER 10\\Picture\\01.png"

