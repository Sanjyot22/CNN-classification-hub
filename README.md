
# Facial Expression Identification 
==================================================================

## Convolution-neural-networks-for-facial-expression-identification
- - - -

+ basic_CNN_development-training_code:
    - createKerasModelArchitectures.py: Creates 12 different CNN model architecture as mentioned my the user
    - runModel.py: Takes all the pre-training arguments from the user and initiates the training
    - trainKerasModel.py: Entire training workflow. This code will start the training w.r.t to user arguments
    - predictionScript.py: This script is used to do prediction using trained weights.

+ basic_CNN_development-predict_on_video
    - affect_net: This directory contains codes to run custom trained categorical model and pre-trained affectnet regression model on a video
    - fer_model: This directory contains code to generated video level predictions using only custom trained categorical model

+ data_preparation
    - Contains pre-processing codes for various datasets like fer-dataset, RAF dataset, Affectnet dataset

+ exp_production_code:
    - data_preprocessing.py: Contains functions w.r.t to data preparation used for exp_detection pipeline 
    - expression_prediction.py: Contains functions w.r.t to model prediction used in exp_detection pipeline 
    - run_exp_detection.py: Code to run expression detection model on input video.
    
+ generative_contrastive_model:
    -WIP

- - - -

## Performance on FER test data 

#### Category identification (Anger, Sad, Neutral, Happy)

![Scheme](https://bitbucket.org/youplus/expression_detection/raw/master/predict_on_video/results_pic.png)


- - - -

## Latency analysis w.r.t to test and business videos 

#### categorical predictions, intensity predictions and video preparation

| (format-size):Duration(secs)        | "Processing time(secs)" | "Processing time(mins)" |
|:-----------------------------------:|:-----------------------:|:-----------------------:|
| (".MOV"-"2.8mb") : 2.6(0.04 mins)   | 87.56                   |1.5                      |
| (".MOV"-"7.0mb") : 4.54(0.08 mins)  | 107.68	                |1.8                      |
| (".MOV"-"8.7mb") : 5.64(0.09 mins)  | 137.13	                |2.3                      |
| (".mp4"-"72mb") : 28.77(0.48 mins)  | 158.06	                |2.6                      |
| (".MOV"- "3.9mb") : 3.7(0.06 mins)  | 384	                |6.4                      |
| (".mp4"-"163mb") : 91.56(1.53 mins) | 973.24	                |16.2                     |

- - - -

*note: All the required files(weights) are present in the path mentioned below:*

- path to emotion model weights
```
https://drive.google.com/file/d/1z-WqfS5RgAWPSwbb0NqdTYnAKtMP9S0j/view?usp=sharing
```
- path to emotion index class mapping dictionary
```
https://drive.google.com/file/d/1EETu__Ua49FvYq9Wk6C4LvY4Ljcpjb-u/view?usp=sharing
```  
- path to excitement-positivity model weights
```
https://drive.google.com/drive/folders/19CJlfD8A2Tkvyvd9ieh7MUVCRPa8Oc59?usp=sharing
```  

*Author: Sanjyot Zade*
