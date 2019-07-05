# CNN classification hub

This project contains 12 different SOTA ready to train CNN architecture's for **image classification**

![CNN classification hub overview](data/cnn-hub-intro.png)

## How to use CNN classification hub

`CNN classification hub` has three major pre-built model training phases

- Model architecture creation
- Model training on specified architecture
- Analyzing best weights from all the saved weights

Aim is to ease the efforts in training different model architecture for image classification tasks. 

**Steps**

1. Create train and test/validation folder. Both the folder should contains images present in separate folder as per each category name.
2. Make the changes to default parameters in "config.py" or run the below command. "config.py" file contains all the model training parameters pre-set for training like model_name_to_train, number of epochs, batch-size etc.

``` 
python run_model.py -dtrain "./data/train" -dvalid "./data/test/" -m vgg16 -w "imagenet" -tt train_all 
```

Note:
- Validation/test dataset is **must** for model training with the hub. 
- Names of category folder in both train and validation/test folder should be exactly same.
- Names of the category folder will be used as tag for that particular class.
- Above commands has only a few required parameters. Description about every dynamic parameter is briefed in "config.py".
- Edit the default parameters in "config.py" or pass them as run-time arguments as above.

## Expected results

Folder named "model_repository" will be created at the location specified (default current folder). Folder structure is mentioned below. 
 
```
model_repository
├── <model name>
│   ├── model_logs
│   |    ├── <weight for each epoch>
│   |    ├── <evaluation csv for each weight>
│   ├── tensor_logs
│   │   training.log
│   │   <model name>_stats.png
│   │   index_class_mapping.json
```
+ A folder with specified model name will be created to store all the model training data.
+ "model_logs" folder keeps all the saved weight after every model training epoch. This folder also has a csv w.r.t to each weight. This is the evaluation csv containing prediction on validation dataset using the corresponding weights. 
+ "tensor_logs" folde keeps all the tensorboard logs.
+ "training.log" file has all the model training statistics for each trained epoch.
+ "<model name>_stats.png" is the graph dislaying model training statistics (both training and testing/validation)
+ "index_class_mapping.json" is a dictionary containing index to label mapping for each category trained. 

**Available models**
```
├── ResNet50
├── Xception
├── VGG16
├── VGG19
├── Inceptionv3
├── InceptionResNetv2
├── NasNetLarge
├── NasNetSmall
├── MobileNet
├── Densenet121
├── Densenet169
└── Densenet201
```

[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/Sanjyot22/CNN-classification-hub)