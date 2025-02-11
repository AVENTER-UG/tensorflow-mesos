from __future__ import print_function

import json
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('WXAgg')



import pathlib
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tfmesos2 import cluster

def main():

    os.environ["MESOS_MASTER"] = "devtest.lab.internal:5050"
    os.environ["MESOS_SSL"] = "true"
    os.environ["MESOS_USERNAME"] = "mesos"
    os.environ["MESOS_PASSWORD"] = "test"

    extra_kw = {}
    extra_kw['fetcher'] = {"http://192.168.150.81:11000/v0/download/flower_photos.tgz":"true"}

    data_dir = pathlib.Path("/mnt/mesos/sandbox/flower_photos/")

    train_ds = tf.keras.utils.image_dataset_from_directory(
             data_dir,
             validation_split=0.2,
             subset="training",
             seed=123,
             image_size=(180, 180),
             batch_size=32,
             shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
             data_dir,
             validation_split=0.2,
             subset="validation",
             seed=123,
             image_size=(180,180),
             batch_size=32,
             shuffle=True
    )

    class_names = train_ds.class_names
    print(class_names)

    num_classes = len(class_names)

    plt.figure(figsize=(10, 10))

    for images, labels in train_ds.take(1):
        for i in range(5):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


    jobs_def = [
        {
            "name": "ps",
            "num":2
        },
        {
            "name": "worker",
            "num":1
        },
    ]

    client_ip = "192.168.150.81"

    with cluster(jobs_def, client_ip=client_ip, **extra_kw) as c:
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": c.cluster_def
        })

        print(os.environ["TF_CONFIG"])

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.repeat()
        val_ds = val_ds.repeat()

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

        with strategy.scope():
             # Modelldefinition
             model = Sequential([
                layers.Rescaling(1./255, input_shape=(180,180,3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes)  # Stelle sicher, dass `num_classes` definiert ist
             ])

             # Optimizer und Metriken
             optimizer = tf.keras.optimizers.Adam()
             accuracy = tf.keras.metrics.Accuracy()

             model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
             )
        # Modellzusammenfassung
        model.summary()

        epochs = 10
        steps_per_epoch = len(train_ds)

        history = model.fit(
           train_ds,
           validation_data=val_ds,
           epochs=epochs,
           steps_per_epoch=steps_per_epoch
        )

        # Modell speichern
        model.save('examples/saved_models/flowers')
        c.shutdown()

if __name__ == '__main__':
    main()
