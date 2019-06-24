# Import keras Libraries\
from custom_model_arhitectures import customModels
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras import applications

class kerasModels():
    """
    Class used to create cnn-architecture available in keras application.
    """

    def __init__(self,model_name,training_type,num_classes):
        """
        Constructor to define parameters for model architecture creation.

        Arguments:
            model_name {str} -- name of the model to be generated for classification.
            training_type {str} -- string representing weather to train complete/partial model.
            num_classes {int} -- number of classes in classification tasks.
        """

        # defining class level variables
        self.MODELNAME = model_name
        self.TRAINING_TYPE = training_type
        self.NUMBER_OF_CLASSES = num_classes
        self.CUSTOM_MODELS = ["micro_exp_net"]

    def __decide_model_name__(self):
        """
        This function checks if the specified model name is valid model-name or not.

        Return:
            model_name_for_call {str} -- name of the function that creates the specified model name.
        """

        # initializing the parameters
        model_name_for_call = "invalid"

        # identify the name of the specified model, if exists.
        if (self.MODELNAME.lower() == 'resnet50'):
            model_name_for_call = 'ResNet50_'
        elif (self.MODELNAME.lower() == 'xception'):
            model_name_for_call = 'Xception_'
        elif (self.MODELNAME.lower() == 'vgg16'):
            model_name_for_call = 'VGG16_'
        elif (self.MODELNAME.lower() == 'vgg19'):
            model_name_for_call = 'VGG19_'
        elif (self.MODELNAME.lower() == 'inceptionv3'):
            model_name_for_call = 'InceptionV3_'
        elif (self.MODELNAME.lower() == 'inceptionresnetv2'):
            model_name_for_call = 'InceptionResNetV2_'
        elif (self.MODELNAME.lower() == 'nasnet_large'):
            model_name_for_call = 'NASNetLarge_'
        elif (self.MODELNAME.lower() == 'nasnet_small'):
            model_name_for_call = 'NASNetsmall_'
        elif (self.MODELNAME.lower() == 'densenet121'):
            model_name_for_call = 'DenseNet121_'
        elif (self.MODELNAME.lower() == 'densenet169'):
            model_name_for_call = 'DenseNet169_'
        elif (self.MODELNAME.lower() == 'densenet201'):
            model_name_for_call = 'DenseNet201_'
        elif (self.MODELNAME.lower() == 'mobilenet'):
            model_name_for_call = 'MobileNet_'
        return model_name_for_call

    def create_model_base(self,number_of_layers_to_freeze=7):
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
            custom_model_creation_obj = customModels()
            model_call = (getattr(custom_model_creation_obj, self.MODELNAME))
            return model_call(nclasses = self.NUMBER_OF_CLASSES)

        # creating model architecture from keras application layer
        # check if the specified model name is valid
        model_name_for_call  = self.__decide_model_name__()

        if self.TRAINING_TYPE == 'freeze':
            # create specified model and freeze first 37 model layers
            # model creation call
            model_call = getattr(self, model_name_for_call)
            model , img_width, img_height = model_call()
            model.layers.pop()
            # check if number_of_layers_to_freeze is more than model layers
            if number_of_layers_to_freeze > len(model.layers): number_of_layers_to_freeze = len(model.layers)
            # freeze model layers
            for layer in model.layers[:number_of_layers_to_freeze]:
                layer.trainable = False
            for layer in model.layers[number_of_layers_to_freeze:]:
                layer.trainable = True

        elif self.TRAINING_TYPE == 'train_all':
            # create specified model and set all model layers as trainable
            # model creation call
            model_call = getattr(self, model_name_for_call)
            model , img_width, img_height = model_call()
            # set all layers to trainable
            model.layers.pop()
            for layer in model.layers:
                layer.trainable = True
        else:
            # create specified model and freeze all the pre-trained layers
            # model call
            model_call = getattr(self, model_name_for_call)
            model , img_width, img_height = model_call()
            # set all layers to non-trainable
            model.layers.pop()
            for layer in model.layers:
                layer.trainable = False

        # stacks extra layers to train on pre-trained model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(512, activation='relu',name="last_layer"))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.NUMBER_OF_CLASSES, activation='softmax'))

        # merge the stacked layers with pre-trained model
        model_final = Model(inputs=model.input, outputs=top_model(model.output))
        return model_final , img_height ,img_width

    def ResNet50_(self):
        """
        This function generates model resnet-50 architecture from keras application.

        Return:
            model_final {keras-model} -- resnet-50 architecture.
            img_height {int} -- input height of the image as per resnet-50.
            img_width {int} -- input width of the image as per resnet-50.
        """
        img_width, img_height = 224, 224
        return  applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def Xception_(self):
        """
        This function generates model xception architecture from keras application.

        Return:
            model_final {keras-model} -- xception architecture.
            img_height {int} -- input height of the image as per xception.
            img_width {int} -- input width of the image as per xception.
        """
        img_width, img_height = 299, 299
        return applications.xception.Xception(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG16_(self):
        """
        This function generates model vgg-16 architecture from keras application.

        Return:
            model_final {keras-model} -- vgg-16 architecture.
            img_height {int} -- input height of the image as per vgg-16.
            img_width {int} -- input width of the image as per vgg-16.
        """
        img_width, img_height = 224, 224
        return applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG19_(self):
        """
        This function generates model vgg-16 architecture from keras application.

        Return:
            model_final {keras-model} -- vgg-19 architecture.
            img_height {int} -- input height of the image as per vgg-19.
            img_width {int} -- input width of the image as per vgg-19.
        """
        img_width, img_height = 224, 224
        return applications.vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def InceptionV3_(self):
        """
        This function generates model inception-v3 architecture from keras application.

        Return:
            model_final {keras-model} -- inception-v3 architecture.
            img_height {int} -- input height of the image as per inception-v3.
            img_width {int} -- input width of the image as per inception-v3.
        """
        img_width, img_height = 299, 299
        return applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3)) , img_width, img_height

    def InceptionResNetV2_(self):
        """
        This function generates model inception-resnet-v2 architecture from keras application.

        Return:
            model_final {keras-model} -- inception-resnet-v2 architecture.
            img_height {int} -- input height of the image as per inception-resnet-v2.
            img_width {int} -- input width of the image as per inception-resnet-v2.
        """
        img_width, img_height = 299, 299
        return applications.inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def NASNetLarge_(self):
        """
        This function generates model nasnet architecture from keras application.

        Return:
            model_final {keras-model} -- nasnet-large architecture.
            img_height {int} -- input height of the image as per nasnet-largenasnet-large.
            img_width {int} -- input width of the image as per nasnet-large.
        """
        img_width, img_height = 331, 331
        return applications.nasnet.NASNetLarge(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def NASNetsmall_(self):
        """
        This function generates model nasnet-small architecture from keras application.

        Return:
            model_final {keras-model} -- nasnet-small architecture.
            img_height {int} -- input height of the image as per nasnet-small.
            img_width {int} -- input width of the image as per nasnet-small.
        """
        img_width, img_height = 224, 224
        return applications.nasnet.NASNetMobile(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet121_(self):
        """
        This function generates model densenet-121 architecture from keras application.

        Return:
            model_final {keras-model} -- densenet-121 architecture.
            img_height {int} -- input height of the image as per densenet-121.
            img_width {int} -- input width of the image as per densenet-121.
        """
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet121(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet201_(self):
        """
        This function generates model densenet-201 architecture from keras application.

        Return:
            model_final {keras-model} -- densenet-201 architecture.
            img_height {int} -- input height of the image as per densenet-201.
            img_width {int} -- input width of the image as per densenet-201.
        """
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet169_(self):
        """
        This function generates model densenet-169 architecture from keras application.

        Return:
            model_final {keras-model} -- densenet-169 architecture.
            img_height {int} -- input height of the image as per densenet-169.
            img_width {int} -- input width of the image as per densenet-169.
        """
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet169(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def MobileNet_(self):
        """
        This function generates model mobilenet architecture from keras application.

        Return:
            model_final {keras-model} -- mobilenet architecture.
            img_height {int} -- input height of the image as per mobilenet.
            img_width {int} -- input width of the image as per mobilenet.
        """
        img_width, img_height = 224, 224
        return applications.mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

