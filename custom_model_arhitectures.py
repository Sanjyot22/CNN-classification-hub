import tensorflow as tf
import tensorflow.keras as keras
from tensor_hub_code.classifiers import SqueezeNet


class CustomModels():
    """
    Class used to create custom cnn-architecture.
    """

    def micro_cnn_net(self, n_classes, img_width, img_height):
        """
        micro_cnn_net creates a two layer cnn architecture for classification.

        Arguments:
            n_classes {int} -- number of layers to freeze in an architecture.
            img_width {int} -- input height of the image as per current architecture
            img_height {int} -- input width of the image as per current architecture
        Return:
            model_final  {keras-model} -- 2 layer CNN architecture.
            img_height {int} -- input height of the image as per current architecture
            img_width {int} -- input width of the image as per current architecture
        """
        # tf.keras.initializers.

        # Weights and baises initializations
        conv_weights_initializations = tf.contrib.layers.xavier_initializer_conv2d()
        bias_initializations = keras.initializers.random_normal(stddev=0.5)
        fully_connected_weights_initialization = tf.contrib.layers.xavier_initializer()


        # stacking cnn architecture
        inputs = keras.Input(shape=(img_width, img_height, 3))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(8,8), activation='relu',padding="same",strides=2,
                                    kernel_initializer=conv_weights_initializations,
                                    bias_initializer=bias_initializations)(inputs)
        pooled1 = keras.layers.MaxPooling2D(pool_size=2,padding="same")(conv1)

        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', padding="same",strides=2,
                                    kernel_initializer=conv_weights_initializations,
                                    bias_initializer=bias_initializations)(pooled1)
        pooled1 = keras.layers.MaxPooling2D(pool_size=2, padding="same")(conv2)

        pooled1_f = keras.layers.Flatten()(pooled1)
        penaltimate_layer_ = keras.layers.Dense(768, activation='relu',
                                                kernel_initializer=fully_connected_weights_initialization,
                                                bias_initializer=bias_initializations)(pooled1_f)
        penaltimate_layer = keras.layers.Dropout(0.5)(penaltimate_layer_)
        output = keras.layers.Dense(n_classes, activation='softmax',
                                    kernel_initializer=fully_connected_weights_initialization,
                                    bias_initializer=bias_initializations)(penaltimate_layer)
        # final model
        model_arch = keras.Model(inputs = inputs , outputs=output)
        return  model_arch, img_width, img_height

    def squeeze_net(self, n_classes, img_width, img_height, fire_nodes):
        """
        micro_cnn_net creates a two layer cnn architecture for classification.

        Arguments:
            nclasses {int} -- number of layers to freeze in an architecture.
            img_width {int} -- input height of the image as per current architecture
            img_height {int} -- input width of the image as per current architecture
            fire_nodes {list} -- nodes representing number of neurons in each fire modules intermediates
        Return:
            model_final  {keras-model} -- 2 layer CNN architecture.
            img_height {int} -- input height of the image as per current architecture
            img_width {int} -- input width of the image as per current architecture
        """
        img_width = 256 if img_width == None else img_width
        img_height = 256 if img_height == None else img_height

        return SqueezeNet(n_classes=n_classes, img_height=img_height,
                          img_width=img_width, fire_nodes=fire_nodes).model(), img_width, img_height
