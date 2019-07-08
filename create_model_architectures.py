import sys
from config import NUM_NODES
from config import NUM_DROPOUTS
from tensor_hub_code.classifiers import VGG16
from tensor_hub_code.classifiers import VGG19
from tensor_hub_code.classifiers import MobileNet
from tensor_hub_code.classifiers import ResNet50
from tensor_hub_code.classifiers import Xception
from custom_model_arhitectures import CustomModels
from tensor_hub_code.classifiers import InceptionV3
from tensor_hub_code.classifiers import DenseNet121
from tensor_hub_code.classifiers import DenseNet169
from tensor_hub_code.classifiers import DenseNet201
from tensor_hub_code.classifiers import NASNetLarge
from tensor_hub_code.classifiers import NASNetMobile
from tensor_hub_code.classifiers import InceptionResNetV2


class Models():
    """
    Class used to create cnn-architecture available in keras application.
    """

    def __init__(self, model_name, training_type, num_classes, height, width, num_freeze_layer):
        """
        Constructor to define parameters for model architecture creation.

        Arguments:
            model_name {str} -- name of the model to be generated for classification.
            training_type {str} -- string representing weather to train complete/partial model.
            num_classes {int} -- number of classes in classification tasks.
            num_freeze_layer {int} -- number of layers to freeze if training type is "freeze_some"
        """

        # defining class level variables
        self.MODELNAME = model_name
        self.HEIGHT = height
        self.WIDTH = width
        self.TRAINING_TYPE = training_type
        self.NUMBER_OF_CLASSES = num_classes
        self.CUSTOM_MODELS = ["micro_exp_net"]
        self.NUMBER_LAYER_FREEZE = num_freeze_layer

    def __model_call__(self, model_name):
        """
        This function checks if the image height/width is specified and create keras model accordingly.

        Arguments:
            model_name {str} -- name of the model to be generated for classification.

        Return:
            model_obj {model-object} -- initiated model creation object.
        """
        if (self.HEIGHT != None) and (self.WIDTH != None):
            model_obj = eval(model_name)(
                n_classes=self.NUMBER_OF_CLASSES, img_height=self.HEIGHT, img_width=self.WIDTH,
                num_nodes=NUM_NODES, dropouts=NUM_DROPOUTS, activation="relu"
            )
        elif self.HEIGHT:
            model_obj = eval(model_name)(
                n_classes=self.NUMBER_OF_CLASSES, img_height=self.HEIGHT, num_nodes=NUM_NODES,
                dropouts=NUM_DROPOUTS, activation="relu"
            )
        elif self.WIDTH:
            model_obj = eval(model_name)(
                n_classes=self.NUMBER_OF_CLASSES, img_width=self.WIDTH, num_nodes=NUM_NODES,
                dropouts=NUM_DROPOUTS, activation="relu"
            )
        else:
            model_obj = eval(model_name)(n_classes=self.NUMBER_OF_CLASSES, num_nodes=NUM_NODES,
                                         dropouts=NUM_DROPOUTS, activation="relu")

        return model_obj


    def __create_model_obj__(self):
        """
        This function checks if the specified model name is valid model-name or not.

        Return:
            model_obj {model-object} -- initiated model creation object.
        """
        # initializing the parameters
        model_obj = "invalid"

        # identify the name of the specified model, if exists.
        if self.MODELNAME.lower() == 'resnet50':
            model_obj = self.__model_call__(model_name="ResNet50")

        elif self.MODELNAME.lower() == 'xception':
            model_obj = self.__model_call__(model_name="Xception")

        elif self.MODELNAME.lower() == 'vgg16':
            model_obj = self.__model_call__(model_name="VGG16")

        elif self.MODELNAME.lower() == 'vgg19':
            model_obj = self.__model_call__(model_name="VGG19")

        elif self.MODELNAME.lower() == 'inceptionv3':
            model_obj = self.__model_call__(model_name="InceptionV3")

        elif self.MODELNAME.lower() == 'inceptionresnetv2':
            model_obj = self.__model_call__(model_name="InceptionResNetV2")

        elif self.MODELNAME.lower() == 'nasnet_large':
            model_obj = self.__model_call__(model_name="NASNetLarge")

        elif self.MODELNAME.lower() == 'nasnet_small':
            model_obj = self.__model_call__(model_name="NASNetMobile")

        elif self.MODELNAME.lower() == 'densenet121':
            model_obj = self.__model_call__(model_name="DenseNet121")

        elif self.MODELNAME.lower() == 'densenet169':
            model_obj = self.__model_call__(model_name="DenseNet169")

        elif self.MODELNAME.lower() == 'densenet201':
            model_obj = self.__model_call__(model_name="DenseNet201")

        elif self.MODELNAME.lower() == 'mobilenet':
            model_obj = self.__model_call__(model_name="MobileNet")

        else:
            print("Model name '{}' not in the list".format(self.MODELNAME))
            sys.exit()
        return model_obj

    def create_model_base(self):
        """
        create_model_base is used to generated spcified architecture, freeze specified
        layers and add the final classification layer.

        Arguments:
            number_of_layers_to_freeze {int} -- number of layers to freeze in an architecture.

        Return:
            model_final  {keras-model} -- final model trainable/pre-trained architecture.
            img_height {int} -- input height of the image as per current architecture
            img_width {int} -- input width of the image as per current architecture
        """

        # creating custom model architecture
        if self.MODELNAME in self.CUSTOM_MODELS:
            custom_model_creation_obj = CustomModels()
            model_call = (getattr(custom_model_creation_obj, self.MODELNAME))
            return model_call(nclasses=self.NUMBER_OF_CLASSES)

        # creating model architecture
        # check if the specified model name is valid
        model_obj = self.__create_model_obj__()
        model_final = getattr(model_obj, "model")()  # creating model architecture from model-object
        img_height = model_final.layers[0].output_shape[0][2]
        img_width = model_final.layers[0].output_shape[0][1]

        if self.TRAINING_TYPE == 'freeze_some':
            # check if number_of_layers_to_freeze is more than model layers
            if self.NUMBER_LAYER_FREEZE > len(model_final.layers):
                number_of_layers_to_freeze = len(model_final.layers)
            else:
                number_of_layers_to_freeze = self.NUMBER_LAYER_FREEZE

            # freeze mentioned model layers
            for layer in model_final.layers[:number_of_layers_to_freeze]:
                layer.trainable = False
            for layer in model_final.layers[number_of_layers_to_freeze:]:
                layer.trainable = True

        elif self.TRAINING_TYPE == 'train_all':
            # set all layers to trainable
            for layer in model_final.layers:
                layer.trainable = True

        elif self.TRAINING_TYPE == 'freeze_all':
            # freeze all the pre-trained layers
            # set all layers to non-trainable
            for layer in model_final.layers:
                layer.trainable = False
        else:
            print("Invalid model training type")
            sys.exit()

        return model_final, img_height, img_width
