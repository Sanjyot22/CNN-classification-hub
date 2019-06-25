# Import keras Libraries
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import  load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers

from time import time
import pandas as pd
import json
import sys

# Import other python Libraries
import os, math
import shutil
from glob import glob
import matplotlib.pyplot as plt


#Import user-defined Libraries
from create_keras_model_architectures import kerasModels
from prediction_script import do_predictions

class kerasModelTraining():
    """
    This class is used to initiate model training as per parameters mentioned in config file.
    """

    def __init__(self,data_dir_train,data_dir_valid,batch_size,epochs,model_name,training_type,save_loc,weights,clear):
        """
        Constructor to define parameters for model training and clear the logs, if specified.

        Arguments:
            data_dir_train {str} -- path to training folder.
            data_dir_valid {str} -- path to validation folder.
            save_loc {str} -- path to save model logs and weights.
            batch_size {int} -- image batch size during model training.
            epochs {int} -- number of epochs the model should train on the data.
            model_name {str} -- name of the model to train.
            training_type {str} -- type of train, as described in config file.
            weights {str} -- whether to pick image-net weights
            clear {boolean} -- whether to clear earlier model training logs and weights.
        """


        # assign all the model training parameters as per config file
        self.TRAIN_DIR = data_dir_train
        self.VALID_DIR = data_dir_valid
        self.SAVE_LOC = save_loc
        self.BATCHSIZE = batch_size
        self.EPOCHS = epochs
        self.MODEL_NAME = model_name
        self.TRAINING_TYPE = training_type
        self.WEIGHTS = weights

        # check the paths are valid
        if not os.path.exists(self.TRAIN_DIR) : print ("Invalid training path");sys.exit();
        if not os.path.exists(self.VALID_DIR) : print ("Invalid validation path");sys.exit();
        if not os.path.exists(self.SAVE_LOC) : print ("Invalid save location");sys.exit();

        # Remove unwanted folders in mac
        remove_unwanted_folders = "find . -name '.DS_Store' -type f -delete"
        os.system(remove_unwanted_folders)
        remove_unwanted_folders = "find {} -name '.DS_Store' -type f -delete".format(data_dir_train)
        os.system(remove_unwanted_folders)
        remove_unwanted_folders = "find {} -name '.DS_Store' -type f -delete".format(data_dir_valid)
        os.system(remove_unwanted_folders)

        # derive other required model training parameters
        # NOTE: It is very important that training and validation folder has only folders with name of the category
        #       class. And no extra folders/files. Names of each class folder in training and validation must be same.
        self.NUMBER_OF_CLASSES = len(os.listdir(data_dir_train))
        self.TRAIN_SAMPLES = sum([len(files) for r, d, files in os.walk(data_dir_train)])
        self.VALIDATION_SAMPLES = sum([len(files) for r, d, files in os.walk(data_dir_train)])

        # list of all the model architecture available in keras applications
        self.keras_models = [
            'xception', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'inceptionresnetv2', 'nasnet_small','nasnet_large',
            'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'micro_exp_net'
        ]

        # clears the earlier model training logs
        if str(clear) == "True":
            self.__clear_logs__(self.MODEL_NAME)

        # define model to be created
        self.model_final = ""

    def __clear_logs__(self,model_name):
        """
        This function delete model logs and weights.

        Arguments:
            model_name {str} -- name of the model whose logs and weights are to be deleted.
        """
        # preserve model initialization wights
        if self.WEIGHTS != "imagenet":
            weights_name = self.WEIGHTS.split("/")[-1]
            os.system("cp {0} /tmp/{1}".format(self.WEIGHTS,weights_name))

        # delete all the model logs as per model name
        model_log_folder = os.path.join(self.SAVE_LOC,'model_repository/'+model_name+'/model_logs/')
        if os.path.exists(model_log_folder):
            shutil.rmtree(model_log_folder)
            os.makedirs(model_log_folder)

        # delete all the tensorboard logs as per model name
        tensor_log_folder = os.path.join(self.SAVE_LOC, 'model_repository/'+model_name+'/tensor_logs/')
        if os.path.exists(tensor_log_folder):
            shutil.rmtree(tensor_log_folder)

        # restore model initialization wights
        if self.WEIGHTS != "imagenet":
            weights_name = self.WEIGHTS.split("/")[-1]
            os.system("cp /tmp/{0} {1}".format(weights_name,self.WEIGHTS))
        return

    def __create_log_folders__(self,model_name):
        """
        This function creates model logs folders.

        Arguments:
            model_name {str} -- name of the model whose logs and weights are to be saved.
        """

        # create folder to save model logs as per model name
        model_log_folder = os.path.join(self.SAVE_LOC,'model_repository/'+model_name+'/model_logs/')
        if not os.path.exists(model_log_folder):
            os.makedirs(model_log_folder)

        # create folder to save tensorboard logs as per model name
        tensor_log_folder = os.path.join(self.SAVE_LOC, 'model_repository/'+model_name+'/tensor_logs/')
        if not os.path.exists(tensor_log_folder):
            os.makedirs(tensor_log_folder)

        return

    def __load_model_arhitecture__(self):
        """
        This function loads the specified model architecture.

        return:
            model_final {keras-model} -- generated keras as per parameters in config.
            img_width {int} -- input width of the image
            img_height {int} -- input height of the image
        """
        print("Number of classes is {}".format(self.NUMBER_OF_CLASSES))

        # getting the model architecture
        modelCreator = kerasModels(self.MODEL_NAME, self.TRAINING_TYPE, self.NUMBER_OF_CLASSES)
        if (self.MODEL_NAME in self.keras_models):
            model_final, img_width, img_height = modelCreator.create_model_base()
        else:
            print("Please specify the model name from the available list")
            print(self.keras_models)
            sys.exit()

        # for training to re-start
        # loading the weights from earlier iteration
        if self.WEIGHTS != "imagenet":
            model_final = load_model(self.WEIGHTS)

        # final model summary
        model_final.summary()
        print("\nModel has {} layers".format(len(model_final.layers)))
        return model_final, img_width, img_height

    def __define_model_compilation__(self):
        """
        This function defines model compilation, keras and custom call_backs.

        Arguments:
            ??


        return:
            call_back_list {list} -- list containing all the callback functions
        """

        # model compilation definitions
        self.model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
                                 metrics=["accuracy"])

        # define callbacks
        call_back_list = []

        # early stopping call-back
        self.early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=8, verbose=1, mode='auto',
                                            restore_best_weights=True)
        call_back_list.append(self.early_stopping)

        # logging stats to a csv file
        self.csv_logger = CSVLogger(os.path.join(self.SAVE_LOC, "model_repository",self.MODEL_NAME,"training.log"))
        call_back_list.append(self.csv_logger)

        # tbCallBack = TensorBoard(log_dir=dir_path + '/model_repository/'+self.MODEL_NAME+'/tensor_logs/' + '/{0}'.format(time()))


        # custom learning rate scheduler
        def step_decay(EPOCH):
            initial_lrate = 0.001
            drop = 0.1
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + EPOCH) / epochs_drop))
            print("\n==== Epoch: {0:} and Learning Rate: {1:} ====".format(EPOCH, lrate))
            return lrate
        self.change_lr = LearningRateScheduler(step_decay)
        call_back_list.append(self.change_lr)

        # weights and logs saver
        # This callback will save the current weights after every epoch
        # The name of weight file contains epoch number, val accuracy
        file_path = os.path.join(self.SAVE_LOC, "model_repository",self.MODEL_NAME,"model_logs","weights-{epoch:02d}-{val_accuracy:.2f}.h5")
        checkpoints = ModelCheckpoint(
            filepath=file_path,  # Path to the destination model file
            # The two arguments below mean that we will not overwrite the
            # model file unless `val_loss` has improved, which
            # allows us to keep the best model every seen during training.
            monitor='val_accuracy',
            save_best_only=False,
        )

        call_back_list.append(checkpoints)
        return call_back_list

    def __prepare_data__(self,img_height, img_width):
        """
        This functionis used to create image data generators for training and validation dataset.

        Arguments:
            img_height {int} -- height of input image
            img_width {int} -- width of input image

        return:
            train_generator {generator} -- train data generator
            valid_generator {generator} -- valid data generator
        """


        # Initiate the train and test generators with data Augumentation
        train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,rotation_range=25)
        test_datagen = ImageDataGenerator( rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(img_height, img_width),
            batch_size=self.BATCHSIZE,
            class_mode="categorical"
        )

        validation_generator = test_datagen.flow_from_directory(
            self.VALID_DIR,
            target_size=(img_height, img_width),
            class_mode="categorical"
        )
        return train_generator, validation_generator

    def __pre_training_report__(self):
        """
        This function reports all the model training stats.
        """
        print()
        print("Pre-training report:")
        print('data_dir_train: ', self.TRAIN_DIR)
        print('data_dir_valid: ', self.VALID_DIR)
        print('save_location', os.path.join(self.SAVE_LOC, "model_repository/"))
        print('model_name: ', self.MODEL_NAME)
        print('# epochs: ', self.EPOCHS)
        print('batch_size:', self.BATCHSIZE)
        print('training_type:', self.TRAINING_TYPE)
        print('weights: ', self.WEIGHTS)
        print()

    def __plot_model_training_history__(self, history_dict, plot_val=True, chart_type="--o"):
        acc = history_dict['accuracy']
        loss = history_dict['loss']

        if plot_val:
            val_acc = history_dict['val_accuracy']
            val_loss = history_dict['val_loss']

        # visualize model training
        epochs = range(1, len(acc) + 1)
        fig, axs = plt.subplots(1, 2,figsize=(15,5))
        axs[0].plot(epochs, loss, chart_type, label='Training loss')
        if plot_val:
            axs[0].plot(epochs, val_loss, chart_type, label='Validation loss')
            axs[0].set_title('training & validation loss')
        else:
            axs[0].set_title('training loss')

        axs[1].plot(epochs, acc, chart_type, label='Training acc')
        if plot_val:
            axs[1].plot(epochs, val_acc, chart_type, label='Validation acc')
            axs[1].set_title('training & validation accuracy')
        else:
            axs[1].set_title('training accuracy')

        plt.show()
        # plt.close()

    def identify_best_validation_weights(self,log_file,how_many):
        training_logs = pd.read_csv(log_file)
        best_weights = training_logs.nlargest(how_many,"val_acc(%)")
        list_of_weights = best_weights["model_path"].tolist()
        return list_of_weights

    def save_final_model_data(self,final_model,generator_for_index_map):
        dir_path = os.path.realpath(os.path.dirname(__file__))
        model_repo = dir_path + '/model_repository/' + self.MODEL_NAME + '/model_logs/'
        pathtomodel = (model_repo + 'final_best_weights.h5')
        final_model.save(pathtomodel)
        # saving model class mapping
        with open(dir_path + '/model_repository/' + self.MODEL_NAME + '/model_logs/' + "Class_Index_Map.json",
                  "w") as write_file:
            json.dump(generator_for_index_map.class_indices, write_file)

    def find_best_weights_from_all_epochs(self,img_height, img_width):

        dir_path = os.path.realpath(os.path.dirname(__file__))

        print ("Starting to analyze best weights on validation data")
        path_to_weights_folder = dir_path +'/model_repository/'+self.MODEL_NAME+'/model_logs/'
        path_validation_images = self.VALID_DIR

        prediction_object = do_predictions(img_height, img_width,self.MODEL_NAME, self.TRAINING_TYPE, self.NUMBER_OF_CLASSES)

        log_file_path = os.path.join(path_to_weights_folder, "training_log.csv")
        weights_to_test = self.identify_best_validation_weights(log_file_path,how_many=8)

        log_file = open(os.path.join(path_to_weights_folder, "actual_validation_logs.txt"), "w+")
        for weight_file_path in weights_to_test:
            weights_name = weight_file_path.split("/")[-1]

            print("Weights : {}".format(weights_name))
            print("Images : {}".format(path_validation_images))
            structured_results, accuracy_, precision_, recall_, accuracy_threshold_ = prediction_object.do_predictions_while_training(
                path_validation_images, weight_file_path, 0)
            log_file.write(
                "Epoc: {} val_acc2: {}% (precision:{},recall:{},threshold:{})\n".format(weights_name, accuracy_,precision_, recall_,
                                                                                        accuracy_threshold_))
        log_file.close()
        print ("Analysis complete")
        return path_to_weights_folder

    def train(self):
        """
        Train function initiates the model training by creating model, preparing data and compiling-evaluating the model.
        """
        # creates the required folder structure
        self.__create_log_folders__(self.MODEL_NAME)

        # constructing model architecture
        self.model_final, img_width, img_height = self.__load_model_arhitecture__()

        # model compilation and definiting all the training call backs
        call_backs = self.__define_model_compilation__()

        # training generator creation - data prep
        train_generator, validation_generator = self.__prepare_data__(img_height=img_height, img_width=img_width)

        # reporting training parameters
        self.__pre_training_report__()

        print ("\nStarted Model training..\n")
        # initiating model training
        validation_steps =  self.VALIDATION_SAMPLES//self.BATCHSIZE
        history = self.model_final.fit_generator(
            train_generator,
            steps_per_epoch=self.TRAIN_SAMPLES//self.BATCHSIZE,
            epochs=self.EPOCHS,verbose=1,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=call_backs
        )
        # saving final model
        self.save_final_model_data(self.model_final,validation_generator)

        # plotting model training history
        self.__plot_model_training_history__(history.history)

        # cleaning model variables
        del self.model_final
        print ("Model Training complete !\n")
        return self.find_best_weights_from_all_epochs(img_height,img_width)



