# import  python Libraries
import os
import sys
import glob
import json
import math
import shutil
import warnings
from config import *
import pandas as pd
import matplotlib.pyplot as plt

# Import keras Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import user-defined libraries
from prediction_script import DoPredictions
from create_model_architectures import Models
warnings.filterwarnings("ignore")


class ModelTraining(Models):

    """
    This class is used to initiate model training as per parameters mentioned in config file.
    """

    def __init__(self, data_dir_train, data_dir_valid, batch_size, epochs, model_name, height,
                 width, training_type, save_loc, weights, start_train, post_eval):
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
        self.TRAIN_DIR = str(data_dir_train)
        self.VALID_DIR = str(data_dir_valid)
        self.SAVE_LOC = str(save_loc)
        self.BATCHSIZE = int(batch_size)
        self.EPOCHS = int(epochs)
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.TRAINING_TYPE = str(training_type)
        self.WEIGHTS = weights
        self.MODEL_NAME = str(model_name)

        # convert to boolean
        self.START_TRAIN = False if str(start_train) == "False" else True
        self.POST_EVALUATION = False if str(post_eval) == "False" else True

        # check the paths are valid
        if not os.path.exists(self.TRAIN_DIR): print ("\nInvalid training path\n"); sys.exit();
        if not os.path.exists(self.VALID_DIR): print ("\nInvalid validation path\n"); sys.exit();
        if not os.path.exists(self.SAVE_LOC): print ("\nInvalid save location\n"); sys.exit();

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
        self.CLASSES = os.listdir(data_dir_train)
        self.TRAIN_SAMPLES = sum([len(files) for r, d, files in os.walk(data_dir_train)])
        self.VALIDATION_SAMPLES = sum([len(files) for r, d, files in os.walk(data_dir_train)])

        # list of all the model architecture available in keras applications
        self.keras_models = [
            'xception', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'inceptionresnetv2', 'nasnet_small','nasnet_large',
            'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'micro_exp_net'
        ]

        # # clears the earlier model training logs
        # if str(clear) == "True":
        #     self.__clear_logs__()

        # define model to be created
        self.model_final = ""

        # Models constructor initialization
        super(ModelTraining, self).__init__(self.MODEL_NAME, self.TRAINING_TYPE, self.NUMBER_OF_CLASSES,
                                            self.IMG_HEIGHT, self.IMG_WIDTH, NUMBER_OF_LAYERS_TO_FREEZE)
        # logs folder name
        if ITERATION_NAME != "":
            self.MODEL_NAME_ = model_name  # This model name is used for architecture definitions parameter
            self.MODEL_NAME = ITERATION_NAME + "_" + model_name  # This model name is used for logs repositories
        else:
            self.MODEL_NAME_ = model_name  # This model name is used for architecture definitions parameter
            self.MODEL_NAME = model_name  # This model name is used for logs repositories

    def __clear_logs__(self):
        """
        This function deletes model-logs and weights.
        """
        # preserve model initialization wights
        if self.WEIGHTS != "imagenet":
            weights_name = self.WEIGHTS.split("/")[-1]
            os.system("cp {0} /tmp/{1}".format(self.WEIGHTS,weights_name))
            weight_folder, weight_name = os.path.split(self.WEIGHTS)

        # delete all the model logs as per model name
        model_log_folder = os.path.join(self.SAVE_LOC,'model_repository', self.MODEL_NAME)
        if os.path.exists(model_log_folder):
            shutil.rmtree(model_log_folder)

        # restore model initialization wights
        if self.WEIGHTS != "imagenet":
            os.makedirs(weight_folder)
            weights_name = self.WEIGHTS.split("/")[-1]
            os.system("cp /tmp/{0} {1}".format(weights_name,self.WEIGHTS))
        return

    def __create_log_folders__(self):
        """
        This function creates model logs folders.
        """

        # create folder to save model logs as per model name
        model_log_folder = os.path.join(self.SAVE_LOC,'model_repository', self.MODEL_NAME, 'model_logs')
        if not os.path.exists(model_log_folder):
            os.makedirs(model_log_folder)
        else:
            if os.listdir(model_log_folder) and (self.START_TRAIN==True):
                print("\nDirectory: {},\nis not empty. Delete and restart".format(model_log_folder))
                print("                  or")
                print("Change 'ITERATION_NAME' parameter in config.py\n")
                sys.exit()

        # create folder to save tensorboard logs as per model name
        tensor_log_folder = os.path.join(self.SAVE_LOC,'model_repository', self.MODEL_NAME, 'tensor_logs')
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
        # getting the model architecture

        if self.MODEL_NAME_ in self.keras_models:
            model_final, img_width, img_height = self.create_model_base()
        else:
            print("Please specify the model name from the available list")
            print(self.keras_models)
            sys.exit()

        # for training to re-start
        # loading the weights from earlier iteration
        if self.WEIGHTS not in ["imagenet", None]:
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
        self.model_final.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

        # define callbacks
        call_back_list = []

        # early stopping call-back
        if EARLY_STOPPING:
            self.early_stopping = EarlyStopping(monitor=ES_MONITOR, min_delta=ES_MIN_DELTA,
                                                patience=ES_PATIENCE, verbose=ES_VERBOSE, mode=ES_MODE,
                                                restore_best_weights=ES_RESTORE_BEST_WEIGHTS)
            call_back_list.append(self.early_stopping)

        # logging stats to a csv file
        self.csv_logger = CSVLogger(os.path.join(self.SAVE_LOC, "model_repository", self.MODEL_NAME, "training.log"))
        call_back_list.append(self.csv_logger)


        # custom learning rate scheduler
        if OPTIMIZER == "sgd":
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
        file_path = os.path.join(
            self.SAVE_LOC, "model_repository", self.MODEL_NAME, "model_logs",
            "weights-{epoch:03d}-{val_accuracy:.4f}.h5"
                                 )
        checkpoints = ModelCheckpoint(
            filepath=file_path,  # Path to the destination model file
            # The two arguments below mean that we will not overwrite the
            # model file unless `val_loss` has improved, which
            # allows us to keep the best model every seen during training.
            monitor=SAVING_METRIC,
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

        # Initiate the train and test generators with data Augmentation
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=HORIZONTAL_FLIP,
                                           rotation_range=ROTATION_RANGE)
        test_datagen = ImageDataGenerator( rescale=1./255)

        print ("\nTraining data:")
        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(img_height, img_width),
            batch_size=self.BATCHSIZE,
            class_mode="categorical"
        )

        print ("\nValidation/test data:")
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
        print('iteration_name: '+ITERATION_NAME)
        print('data_dir_train: ', self.TRAIN_DIR)
        print('data_dir_valid: ', self.VALID_DIR)
        print('save_location:', os.path.join(self.SAVE_LOC, "model_repository/"))
        print("classes: {}".format(self.CLASSES))
        print('model_name: ', self.MODEL_NAME_)
        print('(width,height): ({},{})'.format(self.IMG_WIDTH, self.IMG_HEIGHT))
        print('# epochs: ', self.EPOCHS)
        print('batch_size:', self.BATCHSIZE)
        print('training_type:', self.TRAINING_TYPE)
        print('weights: ', self.WEIGHTS)
        print('start_training: '+str(self.START_TRAIN))
        print('post_evaluation: '+str(POST_EVALUATION))
        print()

    def __plot_model_training_history__(self, history_dict, plot_val=True, chart_type="--o"):
        """
        This function is used to plot training stats and save the graphs to model repository

        Arguments:

        history_dict {dict} -- model training stats
        plot_val {bool} -- weather to plot validation stats in the graphs
        chart_type {str} -- chart type to plot as per matplotlib
        """
        # collecting training stats
        acc = history_dict['accuracy']
        loss = history_dict['loss']
        if plot_val:
            val_acc = history_dict['val_accuracy']
            val_loss = history_dict['val_loss']

        # visualize training stats
        epochs = range(1, len(acc) + 1)
        fig, axs = plt.subplots(1, 2,figsize=(15,5))
        axs[0].plot(epochs, loss, chart_type, label='Training loss')
        if plot_val:
            axs[0].plot(epochs, val_loss, chart_type, label='Validation loss')
            axs[0].set_title('training & validation loss')
        else:
            axs[0].set_title('training loss')

        # visualize validation stats
        axs[1].plot(epochs, acc, chart_type, label='Training acc')
        if plot_val:
            axs[1].plot(epochs, val_acc, chart_type, label='Validation acc')
            axs[1].set_title('training & validation accuracy')
        else:
            axs[1].set_title('training accuracy')

        # save the graph and show
        path_to_save_fig = os.path.join(self.SAVE_LOC, "model_repository",
                                        self.MODEL_NAME, self.MODEL_NAME+"_stats.png")
        plt.savefig(path_to_save_fig)
        if SHOW:
            plt.show()
        plt.close()


    def __save_final_model_data__(self,final_model,generator_for_index_map):
        """
        This function is used to save final model weights and class_index json

        Arguments:

        final_model: final keras model trained as per specified epoch
        generator_for_index_map: validation data generator used to extract index mp json
        """
        #saving final model weights
        model_repo = os.path.join(self.SAVE_LOC, "model_repository", self.MODEL_NAME)
        final_model.save(os.path.join(model_repo, "final_best_weights.h5"))

        # saving model class mapping
        with open(os.path.join(model_repo,"Class_Index_Map.json"),"w") as write_file:
            json.dump(generator_for_index_map.class_indices, write_file)
        return

    def identify_best_validation_weights(self,log_file,how_many):
        """
        This function is used to identify best weights as per validation accuracy during training

        Arguments:
        log_file {str} -- path to file containing training stats
        how_many {int} -- number of weights to be considered for re-evaluation

        Return:
        list_of_epochs {list} -- list epochs number with weight validation accuracy
        """
        training_logs = pd.read_csv(log_file)
        best_weights = training_logs.nlargest(how_many,"val_accuracy")
        list_of_epochs = best_weights["epoch"].tolist()
        return list_of_epochs

    def find_best_weights_from_all_epochs(self,img_height, img_width,
                                          how_many_best_weights=HOW_MANY_WEIGHTS_TO_TEST,
                                          accuracy_threshold=ACCURACY_THRESHOLD):
        """
        This functions is used to identify best validation accuracy weights and re-run this weights on test data.
        This is required because during training the validation accuracy that is broadcasted is with batch norma-
        lization. However, image level accuracy without batch normalization varies a lot. Hence validating best
        weights for later accuracy(accuracy without batch normalization is the practical accuracy).

        Arguments:

        img_height {int} --  input height of the images
        img_width {int} --  input width of the images
        how_many_best_weights {int} -- number of best weights to be picked for re-evaluation
        accuracy_threshold {float} -- accuracy above which prediction will be considered correct
        """
        # variable initialization
        model_folder = os.path.join(self.SAVE_LOC,"model_repository",self.MODEL_NAME)
        log_file_path = os.path.join(model_folder, "training.log")
        path_validation_images = self.VALID_DIR

        # identify weights with best validation accuracy during training
        epochs_to_test = self.identify_best_validation_weights(log_file_path, how_many=how_many_best_weights)
        # print (os.path.join(model_folder,"model_logs", "/*.h5"))
        weight_names = glob.glob1(os.path.join(model_folder, "model_logs"), "*.h5")
        weights_to_test = [
            os.path.join(model_folder,"model_logs",weight_name)
            for epoch in epochs_to_test
            for weight_name in weight_names
            if "-"+"{:03d}".format(epoch+1)+"-" in weight_name
        ]

        # starting re-evaluation
        print("Starting to analyze best weights on validation data")
        prediction_object = DoPredictions(
            img_height,
            img_width,
            self.MODEL_NAME_,
            self.TRAINING_TYPE,
            self.NUMBER_OF_CLASSES
        )

        analysis_df = pd.DataFrame([])
        for weight_file_path in weights_to_test:
            weights_name = str(weight_file_path.split("/")[-1])

            print("\nweights : {}".format(weights_name))
            print("images folder: {}".format(path_validation_images))
            # doing prediction on a weight
            structured_results, stats = prediction_object.do_predictions_while_training(
                path_validation_images, weight_file_path, accuracy_threshold)
            path_to_csv = os.path.join(model_folder, "model_logs", weights_name.replace(".h5", ".csv"))
            structured_results.to_csv(path_to_csv ,index=False)
            print ("Result for weight {} saved to {}\n".format(weights_name, path_to_csv))

            # append to final report
            stats_dict = {
                "weights_name":weights_name,
                "accuracy":stats[0],
                "precision":stats[1],
                "recall":stats[2],
                "confidence_threshold":stats[3]
            }
            analysis_df = analysis_df.append(stats_dict, ignore_index=True)

        # saving the final report
        print("Analysis report:\n")
        path_to_report = os.path.join(model_folder, "analysis_report.csv")
        analysis_df.to_csv(path_to_report, index=False)
        print(analysis_df, "\n")

    def train(self):
        """
        This function initiates the model training by creating model, preparing data and compiling-evaluating the model.
        """
        # creates the required folder structure
        self.__create_log_folders__()

        # constructing model architecture
        self.model_final, img_width, img_height = self.__load_model_arhitecture__()

        # reporting training parameters
        self.__pre_training_report__()

        if self.START_TRAIN:
            # model compilation and definiting all the training call backs
            call_backs = self.__define_model_compilation__()

            # training generator creation - data prep
            train_generator, validation_generator = self.__prepare_data__(img_height=img_height, img_width=img_width)

            print ("\nStarted Model training..\n")
            # initiating model training
            validation_steps = self.VALIDATION_SAMPLES//self.BATCHSIZE
            history = self.model_final.fit_generator(
                train_generator,
                steps_per_epoch=self.TRAIN_SAMPLES//self.BATCHSIZE,
                epochs=self.EPOCHS, verbose=VERBOSE,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=call_backs
            )
            # saving final model
            self.__save_final_model_data__(self.model_final,validation_generator)

            # plotting model training history
            self.__plot_model_training_history__(history.history)

            # cleaning model variables
            del self.model_final
            print ("Model Training complete !\n")

        else:
            print("\nChange 'START_TRAINING=TRUE' to begin training")

        logs_folder = os.path.join(self.SAVE_LOC, 'model_repository', self.MODEL_NAME, 'model_logs')
        if (self.POST_EVALUATION == True) and (len(os.listdir(logs_folder)) != 0):
            self.find_best_weights_from_all_epochs(img_height, img_width)

        return os.path.join(self.SAVE_LOC, "model_repository", self.MODEL_NAME)


