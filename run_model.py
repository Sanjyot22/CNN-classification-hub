import os
import argparse
from config import *
from train_keras_model import kerasModelTraining

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an Image Classification model')

    # description about each parameters is briefed in configuration code "config.py"
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-dtrain', '--data_dir_train', default=TRAIN_DIR, help='Path to Training Data')
    optional.add_argument('-dvalid', '--data_dir_valid', default=VALID_DIR, help='Path to Validation Data')
    optional.add_argument('-sl', '--save_loc',default =SAVE_LOCATION, help='location to save model weights and logs')
    optional.add_argument('-m', '--model_name', default=MODEL_NAME,help = 'Pretrianed model name')
    optional.add_argument('-e', '--epochs', default=EPOCHS, type=int, help = 'Number of epochs')
    optional.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int , help = 'Batch-size')
    optional.add_argument('-clear', '--clear', default=CLEAR_LOGS, help = 'Clear earlier model logs')
    optional.add_argument('-tt', '--training_type',default =TRAINING_TYPE,help = "[train_all,freeze_some,freeze_all]")
    optional.add_argument('-w', '--weights',default =WEIGHTS, help='weights imagenet or custom')

    args = parser.parse_args()
    # create model training object
    training_object = kerasModelTraining(
        args.data_dir_train,
        args.data_dir_valid,
        args.batch_size,
        args.epochs,
        args.model_name,
        args.training_type,
        args.save_loc,
        args.weights,
        args.clear
    )
    output_string = training_object.train()


