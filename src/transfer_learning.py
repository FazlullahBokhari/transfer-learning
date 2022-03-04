import argparse
import os
import shutil
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf


STAGE = "transferlearning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    # load Base model
    base_model_path = os.path.join("artifacts", "models/base_models.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.summary()

    ## freeze weights

    for layer in base_model.layers[: -1]:
        print(f"before freezing weights {layer.name}: {layer.trainable}")
        layer.trainable = False
        print(f"after freezing weights {layer.name}: {layer.trainable}")

    ## modify last layer for our problem statements

    base_layers = base_model.layers[: -1]

    new_model = tf.keras.models.Sequential(base_layers)

    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="output_layer")
    )

    new_model.summary()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e