from __future__ import print_function

import json
import os
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tfmesos2 import cluster

# MESOS-Umgebungsvariablen setzen
os.environ["MESOS_MASTER"] = <MESOS_MASTER>:5050"
os.environ["MESOS_SSL"] = "true"
os.environ["MESOS_USERNAME"] = "<MESOS_USERNAME>"
os.environ["MESOS_PASSWORD"] = "<MESOS_PASSWORD>"

extra_kw = {}
extra_kw['fetcher'] = {"http://<DEVELOPERS_IP>:11000/v0/download/flower_photos.tgz": "true"}

data_dir = pathlib.Path("/mnt/mesos/sandbox/flower_photos/")

# Bildverarbeitung und Modelltraining
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    jobs_def = [
        {"name": "ps", "num": 1},
        {"name": "worker", "num": 2},
    ]

    client_ip = "<DEVELOPERS_IP>"

    with cluster(jobs_def, client_ip=client_ip, **extra_kw) as c:
        os.environ["TF_CONFIG"] = json.dumps({"cluster": c.cluster_def})
        print(os.environ["TF_CONFIG"])

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

        batch_size = 1
        img_size = (180, 180)
        num_classes = 5

        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=123,
            shuffle=True)

        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=123,
            shuffle=True)

        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        validation_steps = tf.data.experimental.cardinality(val_ds).numpy()

        print(f"Train batches: {steps_per_epoch}, Val batches: {validation_steps}")


        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)

        with strategy.scope():
            model = create_model(input_shape=(180, 180, 3), num_classes=num_classes)

        model.summary()

        model.fit(train_ds, validation_data=val_ds, epochs=1, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        print(">>>>>>Modelltraining abgeschlossen<<<<<<<<")

        model.save("saved_models/flower_model.keras")
        print(">>>>>>Modell gespeichert.<<<<<<<<")

if __name__ == '__main__':
    main()

