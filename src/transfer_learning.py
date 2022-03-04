import argparse
import os
import shutil
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np


STAGE = "transferlearning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_odd_even_label(labels):
    for idx, label in enumerate(labels):
        labels[idx] = np.where(label%2 == 0, 1, 0)

    return labels    

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

    ## get the data

    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    X_train_full = X_train_full /255.0
    X_test = x_test / 255.

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    y_train_bin, y_test_bin, y_valid_bin = update_odd_even_label([y_train, y_test, y_valid])

    new_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])

    history = new_model.fit(X_train, y_train_bin,
                        epochs=10,
                        validation_data=(X_valid, y_valid_bin), verbose=2)
    
    new_model.evaluate(X_test, y_test_bin)
    new_model_path = os.path.join("artifacts", "models", "new_model.h5")
    new_model.save(new_model_path)

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