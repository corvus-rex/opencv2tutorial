import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gtts import gTTS
import os

def videocap(path):
    cv2.namedWindow("preview") #new cv2 window
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read() #read boolean camera status and frame of VideoCapture
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite(path, frame) #save frame when 'y' is pressed
            plt.imshow(grayresize(path), cmap="gray") #show grey re-scaled image
            plt.show()
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break #stop capture when 'q' is pressed
    cap.release()
    cv2.destroyWindow("preview")

def grayresize(path):
    IMG_SIZE = 28
    image = cv2.imread(path)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to grayscale
    grayresizedimg = cv2.resize(grayimage, (IMG_SIZE, IMG_SIZE)) #resize image to constant IMG_SIZE
    cv2.imwrite(path, grayresizedimg) #save resized image
    return grayresizedimg

def texttospeech(number): #read the predicted number as a text-to-speech mp3 file
    number = str(number)
    text = "Your number is " + number
    language = 'en'
    speech = gTTS(text=text, lang=language, slow=False)
    speech.save("tts.mp3")
    os.system('start tts.mp3')

if __name__ == "__main__":
    IMG_SIZE = 28
    path = 'vidcap/capture0.png' #save captured image to constant file path
    videocap(path) #capture image from webcam into path
    mnist_classifier = tf.keras.models.load_model('mnist_classifier.model') #load model
    preparedTensor = grayresize(path).reshape(-1, IMG_SIZE, IMG_SIZE) #reshape the tensor
    predictedTensor = tf.keras.utils.normalize(~preparedTensor, axis=1) #normalize the INVERTED tensor
    prediction = mnist_classifier.predict(predictedTensor) #predict tensor
    result = np.argmax(prediction) #predicted number as argmax classification
    print(predictedTensor)
    print(result) #print classification
    texttospeech(result) #read the predicted classification aloud