import tensorflow as tf
import keras

class customModels():


    def __init__(self):
        pass

    def micro_exp_net(self, nclasses,img_width_ =84, img_height_ =84):


        # Weights and baises initializations
        conv_weights_initializations = tf.contrib.layers.xavier_initializer_conv2d()
        bias_initializations = keras.initializers.random_normal(stddev=0.5)
        fully_connected_weights_initialization = tf.contrib.layers.xavier_initializer()


        # micro expression architecture for emotion  recognition
        inputs = keras.Input(shape=(img_width_,img_height_,3))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(8,8), activation='relu',padding="same",strides=2,

                                       kernel_initializer=conv_weights_initializations,
                                       bias_initializer=bias_initializations)(inputs)

        pooled1 = keras.layers.MaxPooling2D(pool_size=2,padding="same")(conv1)

        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', padding="same",strides=2,
                                       kernel_initializer=conv_weights_initializations,
                                       bias_initializer=bias_initializations)(pooled1)


        pooled1 = keras.layers.MaxPooling2D(pool_size=2,padding="same")(conv2)
        pooled1_f = keras.layers.Flatten()(pooled1)

        penaltimate_layer_ = keras.layers.Dense(768, activation='relu',
                                                   kernel_initializer=fully_connected_weights_initialization,
                                                   bias_initializer=bias_initializations)(pooled1_f)

        penaltimate_layer = keras.layers.Dropout(0.5)(penaltimate_layer_)

        output = keras.layers.Dense(nclasses, activation='softmax',
                                       kernel_initializer=fully_connected_weights_initialization,
                                       bias_initializer=bias_initializations)(penaltimate_layer)


        model_arch = keras.Model(inputs = inputs , outputs=output)

        # model summary
        # print (model_arch.summary())

        return  model_arch, img_width_, img_height_


