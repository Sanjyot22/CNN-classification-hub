# Import keras Libraries
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from customModelArhitectures  import customModels

class kerasModels():

    def __init__(self,model_name,training_type,num_classes):

        self.MODELNAME = model_name
        self.TRAINING_TYPE = training_type
        self.NUMBER_OF_CLASSES = num_classes
        self.CUSTOM_MODELS = ["micro_exp_net"]

    def decideModelName(self):
        model_name_for_call = "invalid"

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

    def createModelBase(self):

        if self.MODELNAME in self.CUSTOM_MODELS:
            custom_model_creation_obj = customModels()
            modelCall = (getattr(custom_model_creation_obj, self.MODELNAME))
            return  modelCall(nclasses = self.NUMBER_OF_CLASSES)


        model_name_for_call  = self.decideModelName()
        if (self.TRAINING_TYPE == 'freeze'):

            modelCall = getattr(self, model_name_for_call)


            model , img_width, img_height = modelCall()

            model.layers.pop()
            for layer in model.layers[:37]:
                layer.trainable = False
            for layer in model.layers[37:]:
                layer.trainable = True

        if (self.TRAINING_TYPE == 'train_all'):
            modelCall = getattr(self, model_name_for_call)
            model , img_width, img_height = modelCall()

            model.layers.pop()
            for layer in model.layers:
                layer.trainable = True
        else:
            modelCall = getattr(self, model_name_for_call)

            model , img_width, img_height = modelCall()

            model.layers.pop()
            for layer in model.layers:
                layer.trainable = False
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(512, activation='relu',name="last_layer"))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.NUMBER_OF_CLASSES, activation='softmax'))

        model_final = Model(inputs=model.input, outputs=top_model(model.output))
        return model_final , img_height ,img_width

    def ResNet50_(self):
        img_width, img_height = 224, 224
        return  applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def Xception_(self):
        img_width, img_height = 299, 299
        return applications.xception.Xception(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG16_(self):
        img_width, img_height = 224, 224
        return applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG19_(self):
        img_width, img_height = 224, 224
        return applications.vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def InceptionV3_(self):
        img_width, img_height = 299, 299
        return applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3)) , img_width, img_height

    def InceptionResNetV2_(self):
        img_width, img_height = 299, 299
        return applications.inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def NASNetLarge_(self):
        img_width, img_height = 331, 331
        return applications.nasnet.NASNetLarge(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def NASNetsmall_(self):
        img_width, img_height = 224, 224
        return applications.nasnet.NASNetMobile(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet121_(self):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet121(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet201_(self):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet169_(self):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet169(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def MobileNet_(self):
        img_width, img_height = 224, 224
        return applications.mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

