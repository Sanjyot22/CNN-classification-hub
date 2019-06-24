# This scripts contains all the default training parametes pre-set for model training.
# Description about all the required parameters for training is mentioned below.


"""
All the recognition models available:

keras_models= ['xception', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3',
                'inceptionresnetv2', 'nasnet_small','nasnet_large',
                'densenet121', 'densenet169', 'densenet201', 'mobilenet']
"""


# path to training and validation directory
TRAIN_DIR = "./data/training_data" # path to training directory. Classes wise folders of images
VALID_DIR = "./data/validation_data" # path to validation directory. Classes wise folders of images

# Initializing model parameters
MODEL_NAME = "resnet50" # specify the model name from above list
EPOCHS = 100   # number of epochs to train
BATCH_SIZE = 32 # batch size of training
CLEAR_LOGS = True # if True will clear model training logs & weights from the folder
# Note: if re-starting model training the make sure that earlier weights are preserved by keeping "CLEAR_LOGS=False"
# Note: when re-starting model training with "CLEAR_LOGS=True".
#       1. If restarting from earlier saved weights those weights will not be deleted, everything else will be deleted
#       2. If restarting model training from beginning then everything in the folder will be deleted.
# Note: Model logs are only delete as per model name. If training a model with new name & "CLEAR_LOGS=True" then
#       then weights & logs other earlier model name will not be deleted.

# There are three modes of training available:
# ["train_all","freeze_some","freeze_all"]
"""
Model is created by using architecture from keras application model and adding 2 extra dense layer on top of it. 

1. "train_all":
    This mode trains all the  model layers as well as added 2 dense layers. 
 
2. "freeze_some":
    This mode will freeze some of the layers in model(number of layers to freeze can be specified). 
    Rest of the  model layers and  2 dense layers will be trained. 

3. "freeze_all":
    None of the model layers will be trained. Only added last 2 dense layers will be trained.
"""
TRAINING_TYPE = "train_all"

# if "imagenet" then imagenet trained weight intialization is used for keras model.
# However, in order to re-start training weights path can be specified here.
WEIGHTS = "imagenet"
SAVE_LOCATION = "./" # path to folder where model weights and logs will be saved


