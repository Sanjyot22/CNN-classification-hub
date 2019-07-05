import argparse
from config import *
from train_models import ModelTraining

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an Image Classification model')

    # description about each parameters is briefed in configuration code "config.py"
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-dtrain', '--data_dir_train', default=TRAIN_DIR, help='Path to Training Data')
    optional.add_argument('-dvalid', '--data_dir_valid', default=VALID_DIR, help='Path to Validation Data')
    optional.add_argument('-sl', '--save_loc', default=SAVE_LOCATION, help='location to save model weights and logs')
    optional.add_argument('-m', '--model_name', default=MODEL_NAME, help='Pretrianed model name')
    optional.add_argument('-height', '--height', default=IMG_HEIGHT, help='height of the input image to model')
    optional.add_argument('-width', '--width', default=IMG_WIDTH, help='width of the input image to model')
    optional.add_argument('-e', '--epochs', default=EPOCHS, type=int, help='Number of epochs')
    optional.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help = 'Batch-size')
    optional.add_argument('-clear', '--clear', default=CLEAR_LOGS, help='Clear earlier model logs')
    optional.add_argument('-tt', '--training_type', default=TRAINING_TYPE, help="[train_all,freeze_some,freeze_all]")
    optional.add_argument('-w', '--weights', default=WEIGHTS, help='weights imagenet or custom')
    optional.add_argument('-start', '--start_train', default=True, help='bool referring to start model training')

    args = parser.parse_args()
    # create model training object
    training_object = ModelTraining(
        args.data_dir_train,
        args.data_dir_valid,
        args.batch_size,
        args.epochs,
        args.model_name,
        args.height,
        args.width,
        args.training_type,
        args.save_loc,
        args.weights,
        args.clear,
        args.start_train
    )
    output_string = training_object.train()
