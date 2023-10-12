from __future__ import print_function

import tensorflow as tf
import json
import os
from tfmesos2 import cluster


def main():
    jobs_def = [
        {
            "name": "ps",
            "num": 2
        },
        {
            "name": "worker",
            "num":1 
        },
    ]

    with cluster(jobs_def, quiet=False) as c:
        # Cluster-Definition in der TF_CONFIG-Umgebungsvariable festlegen
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": c.cluster_def,
            "task": {
                "type": "worker",
                "index": 0
            }
        })

        # Cluster-Resolver erstellen
        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        # Erstellen einer verteilten Strategie
        strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

        # Konstante auf jedem Server erstellen
        with strategy.scope():
            constant_a = tf.constant(10)
            constant_b = tf.constant(32)

        # Berechnung auf einem anderen Server durchführen
        with tf.device("/job:worker/task:0"):
            op = constant_a + constant_b

        # Berechnung in einer verteilten Strategy-Sitzung ausführen
        with strategy.scope():
            result = op.numpy()
            print("Result is: ")
            print(result)
            c.stop()

if __name__ == '__main__':
    main()
