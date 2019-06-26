# import the dependencies or libraries
import os
import gc
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from PIL import ImageOps
from pandas_ml import ConfusionMatrix
from tensorflow.keras.models import load_model
from create_keras_model_architectures import kerasModels

class do_predictions():

    """
    This class is used to do image level prediction on validation data post model training.
    Same code can be used for inferencing with trained weights, for pratical predictions.
    """

    def __init__(self,img_height,img_width,model_name,training_type,nclasses):
        """
        Constructor to define parameters for model prediction.

        Arguments:

        img_height: {int} -- input height of the image
        img_width: {int} -- input width of the image
        model_name: {str} -- name of the model to do prediction
        training_type: {str} -- types of training used for prediction
        nclasses: {int} -- number of classes used in prediction
        """
        # initialing model parameters
        self.height = img_height
        self.width = img_width
        self.MODEL_NAME = model_name
        self.TRAINING_TYPE = training_type
        self.NUMBER_OF_CLASSES = nclasses

    def load_model_h5(self,hdf5_filename):
        """
        This function is used to load the model.

        Arguments:
        hdf5_filename: {str} -- path to weights file

        Return:
            model {keras-model} -- loaded classification model
        """
        try:
            # loading model from weights directly
            model = load_model(hdf5_filename)
        except:
            print("""
            Model could not be loaded directly. 
            Creating architecture separately and loading
            """)
            # creating the architecture and loading weights
            model , w , h = kerasModels(self.MODEL_NAME, self.TRAINING_TYPE, self.NUMBER_OF_CLASSES).create_model_base()
            model.load(hdf5_filename)
        return model

    def load_labels(self,path_to_datasets_with_labels_folders):
        """
        creates class labels used in training from validation directory.

        Arguments:
        path_to_datasets_with_labels_folders: {str} -- absolute path to the dataset

        Returns: sorted list of class labels
        """
        return sorted(os.listdir(path_to_datasets_with_labels_folders))

    def run_model(self,input_data, labels, model):
        """
        This function is used to do model predictions.

        Arguments:
        input_data: {path/paths} -- path/list_of_paths of input image/images
        labels {list} -- sorted list of class labels
        model {keras-model} -- model trained on training data

        Returns:
        label {str} -- class prediction/predictions
        condidence {float} -- confidence of prediction/predictions
        """
        # doing model prediction
        yhat = model.predict(input_data)
        confidence = max(yhat[0])
        index = np.argmax(yhat)
        label = labels[index]
        return label, confidence

    def load_image_file(self,image_filename):
        """
        This function is used to load input image for model prediction.

        Arguments:
        image_filename: {path} -- path to input image

        Returns:
        image_data: {PIL-object} -- image data loaded in PIL
        """
        # loading and normalizing image file
        new_img = Image.open(image_filename)
        test_img = ImageOps.fit(new_img, (self.height, self.width), Image.ANTIALIAS).convert('RGB')
        test_img = np.array(test_img)
        test_img = test_img.astype(float)
        image_data = test_img.reshape(1, self.height, self.width, 3) / 255
        return image_data

    def calculate_accuracy(self,results,annotations,confidences,accuracy_threshold):
        """
        This function is used to calculate accuracy, precision and recall.

        Arguments:
        results: {list} -- list containing predicion for all the images
        annotations: {list} -- list containing annotationd for all the images
        confidences: {list} -- list containing confidences for all the predictions
        accuracy_threshold: {int} -- accuracy threshold for precision/recall calculation

        Returns:
        accuracy: {float} -- calculated accuracy for the given predictions w.r.t annotations
        precision: {float} -- calculated precision for the given predictions w.r.t annotations
        recall: {float} -- calculated recall for the given predictions w.r.t annotations
        accuracy_threshold: {int} -- accuracy threshold used for precision/recall calculation
        """
        total_entires = len(results)
        correct_count = 0
        correct_count_ = 0
        above_threshold_count = 0

        # calculating correcting prediction w.r.t annotations
        for entry_num, pred in enumerate(results):
            if confidences[entry_num] >= accuracy_threshold:
                above_threshold_count = above_threshold_count + 1

                if pred == annotations[entry_num]:
                    correct_count = correct_count  + 1

            if pred == annotations[entry_num]:
                correct_count_ = correct_count_ + 1

        # calculating precision, recall and accuracy using the above calculations
        if (total_entires > 0):
            accuracy = round((float(correct_count_)/(total_entires))*100,2 )
        else:
            accuracy = 0

        if (above_threshold_count>0):
            precision = round((float(correct_count) / (above_threshold_count))*100,2)
        else:
            precision = 0

        if total_entires >0:
            recall = round((float(above_threshold_count) / total_entires)*100,2)
        else:
            recall = 0

        print ("Overall accuracy is {}%".format(accuracy))
        print ("Precision : {}% Recall : {}% Confidence threshold : {}%".format(precision,recall,accuracy_threshold))
        print (ConfusionMatrix(annotations,results))
        return accuracy,precision,recall,accuracy_threshold

    def do_predictions_while_training(self,input_image_dir_path,weights_file_path,threshold = 0):
        """
        This function is used initiate model predictions w.r.t given weights.

        Arguments:
        input_image_dir_path {str} -- path to input images directory
        weights_file_path {str} -- path to model weights file
        threshold {int} -- accuracy threshold for precision/recall calculation

        Returns:
        df_result {dataframe} -- pandas dataframe containing analysis report w.r.t input weights
        (accuracy,precision,recall, accuracy_threshold) {str} -- tuple containing precision, recall and accuracy
        """

        # loading th model
        print ("Loading model...".format(weights_file_path.split("/")[-1]))
        model = self.load_model_h5(weights_file_path)
        # print ("Model loading complete!\n")

        # removing unwanted folders in mac
        remove_unwanted_folders = "find . -name '.DS_Store' -type f -delete"
        os.system(remove_unwanted_folders)
        remove_unwanted_folders = "find {} -name '.DS_Store' -type f -delete".format(input_image_dir_path)
        os.system(remove_unwanted_folders)

        # loading the class labels
        labels = self.load_labels(input_image_dir_path)
        print ("Detected labels : {}".format(labels))

        # variable initialization and model predictions
        image_names = []
        image_paths = []
        image_annotations = []
        image_confidences = []
        image_predictions = []
        print ("Starting prediction on test images...")
        for image_path in tqdm(glob.glob(input_image_dir_path + '/**/*')):

            # laoding an image
            test_img = self.load_image_file(image_path)

            # model prediction
            prediction = model.predict(test_img)
            prediction = list(prediction[0])

            # saving prediction, annotation and confidences
            max_index =  prediction.index(max(prediction))
            labled_pred = labels[max_index]
            prediction_confidence = max(prediction)
            actual_annotation = image_path.split("/")[-2]
            image_names.append(image_path.split("/")[-1])
            image_paths.append(image_path)
            image_annotations.append(actual_annotation)
            image_confidences.append(prediction_confidence)
            image_predictions.append(labled_pred)

        # calculating accuracies
        accuracy, precision, recall, accuracy_threshold =self.calculate_accuracy(image_predictions, image_annotations, image_confidences, threshold)
        # restructuring prediction, annotation and confidences
        df_result = pd.DataFrame(list(zip(image_paths,image_names,image_annotations,image_predictions,image_confidences)),
                         columns=['image_path', 'Image_name', 'Actual_annotation', 'Image_prediction','Image_confidence' ])
        # cleaning model variables
        del model
        for i in range(3):
            gc.collect()
        return df_result, (accuracy,precision,recall, accuracy_threshold)

