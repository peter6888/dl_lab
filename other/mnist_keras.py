import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Flatten, Dense

def homework_2():
    '''
    Develop a simple neural network using MNIST datasets.
    '''
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    k_size, p_size = (3,3), (2,2)
    # The sub parallel layers
    subInput = tf.keras.Input(shape=(28, 28, 32))
    conv2_1_output = Conv2D(filters = 64, padding="same", kernel_size=k_size, activation=tf.nn.relu)(subInput)
    conv2_1_pooled = MaxPooling2D(pool_size=p_size)(conv2_1_output) # shape = (14, 14, 64)
    conv3_1_output = Conv2D(filters = 256, padding="same", kernel_size=k_size, activation=tf.nn.relu)(conv2_1_pooled)
    conv3_1_pooled = MaxPooling2D(pool_size=p_size)(conv3_1_output)
    conv2_2_output = Conv2D(filters = 64, padding="same", kernel_size=k_size, activation=tf.nn.relu)(subInput)
    conv2_2_pooled = MaxPooling2D(pool_size=p_size)(conv2_2_output) # shape = (14, 14, 64)
    conv3_2_output = Conv2D(filters = 256, padding="same", kernel_size=k_size, activation=tf.nn.relu)(conv2_2_pooled)
    conv3_2_pooled = MaxPooling2D(pool_size=p_size)(conv3_2_output)
    subOutput = concatenate(inputs = [conv3_1_pooled, conv3_2_pooled]) #shape = (7, 7, 512)

    subModel = tf.keras.Model(subInput, subOutput)
    print(subModel.summary())

    model = tf.keras.models.Sequential([
        Conv2D(filters = 32, kernel_size=k_size, padding="same", input_shape=input_shape, activation=tf.nn.relu),
        subModel,
        Flatten(),
        Dense(1000, activation=tf.nn.relu),
        Dense(500, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)

if __name__=="__main__":
    homework_2()

'''--sample output---
(.env) stonepeter@cntk-ubuntu:~/notebooks/dl_lab/other$ python mnist_keras.py 
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 28, 28, 32)   0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 28, 28, 64)   18496       input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 28, 28, 64)   18496       input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 14, 14, 64)   0           conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 14, 14, 64)   0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 14, 14, 256)  147712      max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 14, 14, 256)  147712      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 7, 7, 256)    0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 7, 7, 256)    0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 7, 7, 512)    0           max_pooling2d_1[0][0]            
                                                                 max_pooling2d_3[0][0]            
==================================================================================================
Total params: 332,416
Trainable params: 332,416
Non-trainable params: 0
__________________________________________________________________________________________________
None
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
model (Model)                (None, 7, 7, 512)         332416    
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense (Dense)                (None, 1000)              25089000  
_________________________________________________________________
dense_1 (Dense)              (None, 500)               500500    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5010      
=================================================================
Total params: 25,927,246
Trainable params: 25,927,246
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/1
2018-10-10 17:14:07.431572: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
60000/60000 [==============================] - 551s 9ms/step - loss: 0.1050 - acc: 0.9680
10000/10000 [==============================] - 17s 2ms/step
(.env) stonepeter@cntk-ubuntu:~/notebooks/dl_lab/other$ 
'''
