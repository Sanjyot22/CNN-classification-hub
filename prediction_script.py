# import the dependencies or libraries
from createKerasModelArchitectures import kerasModels
from keras.models import Model, load_model
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm
from pandas_ml import ConfusionMatrix
import numpy as np
import glob
import os,gc


class do_predictions():

    def __init__(self,img_height,img_width,model_name,training_type,nclasses):
        print ("Object of class do prediction created")
        self.height = img_height
        self.width = img_width
        self.MODEL_NAME = model_name
        self.TRAINING_TYPE = training_type
        self.NUMBER_OF_CLASSES = nclasses

    def load_model_h5(self,hdf5_filename):

        try:
            model = load_model(hdf5_filename)
        except:
            print("""
            Model could not be loaded directly. 
            Creating architecture separately and loading
            """)
            model , w , h = kerasModels(self.MODEL_NAME, self.TRAINING_TYPE, self.NUMBER_OF_CLASSES).createModelBase()
        return model

    def load_labels(self,path_to_datasets_with_labels_folders):
        return sorted(os.listdir(path_to_datasets_with_labels_folders))

    def run_model(self,input_data, labels, model):
        yhat = model.predict(input_data)
        confidence = max(yhat[0])
        index = np.argmax(yhat)
        label = labels[index]
        return label, confidence

    def load_image_file(self,image_filename):
        new_img = Image.open(image_filename)
        test_img = ImageOps.fit(new_img, (self.height, self.width), Image.ANTIALIAS).convert('RGB')
        test_img = np.array(test_img)
        test_img = test_img.astype(float)
        image_data = test_img.reshape(1, self.height, self.width, 3) / 255
        return image_data

    def calculate_accuracy(self,results,annotations,confidences,accuracy_threshold):

        total_entires = len(results)
        correct_count = 0
        correct_count_ = 0
        above_threshold_count = 0


        for entry_num, pred in enumerate(results):
            if confidences[entry_num] >= accuracy_threshold:
                above_threshold_count = above_threshold_count + 1

                if pred == annotations[entry_num]:
                    correct_count = correct_count  + 1

            if pred == annotations[entry_num]:
                correct_count_ = correct_count_ + 1

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
        print ("Precision : {}% Recall : {}% Confidence threshold : {}%\n\n".format(precision,recall,accuracy_threshold))
        print (ConfusionMatrix(annotations,results))
        print ("\n")
        return accuracy,precision,recall,accuracy_threshold

    def do_predictions_while_training(self,input_image_dir_path,weights_file_path,threshold = 0):

        # 1.load model
        print ("Loading latest model")
        model = self.load_model_h5(weights_file_path)
        print ("Model loading complete!")

        # 2. Remove unwanted folders
        remove_unwanted_folders = "find . -name '.DS_Store' -type f -delete"
        os.system(remove_unwanted_folders)
        remove_unwanted_folders = "find {} -name '.DS_Store' -type f -delete".format(input_image_dir_path)
        os.system(remove_unwanted_folders)

        # 3.load labels
        labels = self.load_labels(input_image_dir_path)
        print ("Detected labels : {}".format(labels))


        image_names = []
        image_paths = []
        image_annotations = []
        image_confidences = []
        image_predictions = []

        print ("Starting prediction on test images...")
        for image_path in tqdm(glob.glob(input_image_dir_path + '/**/*')):

            test_img = self.load_image_file(image_path)

            prediction = model.predict(test_img)
            prediction = list(prediction[0])

            max_index =  prediction.index(max(prediction))
            labled_pred = labels[max_index]
            prediction_confidence = max(prediction)
            actual_annotation = image_path.split("/")[-2]


            image_names.append(image_path.split("/")[-1])
            image_paths.append(image_path)
            image_annotations.append(actual_annotation)
            image_confidences.append(prediction_confidence)
            image_predictions.append(labled_pred)


        accuracy, precision, recall, accuracy_threshold =self.calculate_accuracy(image_predictions, image_annotations, image_confidences, threshold)


        df_result = pd.DataFrame(list(zip(image_paths,image_names,image_annotations,image_predictions,image_confidences)),
                         columns=['image_path', 'Image_name', 'Actual_annotation', 'Image_prediction','Image_confidence' ])

        del model
        for i in range(3):
            gc.collect()
        return  df_result , accuracy,precision,recall, accuracy_threshold

