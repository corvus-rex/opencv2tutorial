import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def train(x_train, label_train, epoch_num):
    # Neural Network Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) #Transform image into 1D array
    model.add(tf.keras.layers.Dense(420, activation=tf.nn.relu)) #Hidden layer 1 with 420 neurons
    model.add(tf.keras.layers.Dense(420, activation=tf.nn.relu)) #Hidden layer 2 with 420 neurons
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #Probability distribution of label '0' to '9'

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #compile model
    model.fit(x_train, label_train, epochs=epoch_num) #train model
    model.save('mnist_classifier.model') #save model


def predict(x_test):
    mnist_classifier = tf.keras.models.load_model('mnist_classifier.model')
    prediction = mnist_classifier.predict(x_test[1:2])
    print(np.argmax(prediction))

    plt.imshow(x_test[1], cmap = plt.cm.binary)
    plt.show()

def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, label_train), (x_test, label_test) = mnist.load_data()  # set loaded data as training and test var
    x_train = tf.keras.utils.normalize(x_train, axis=1)  # Normalize training data
    x_test = tf.keras.utils.normalize(x_test, axis=1)  # Normalize test data
    print(x_test[1])
    print(x_test[1].shape)

    #train(x_train, label_train, 10)
    predict(x_test)

main()

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()