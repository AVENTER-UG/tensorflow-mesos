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

    with cluster(jobs_def) as c:
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": c.cluster_def,
            "task": {
                "type": "worker",
                "index": 0
            }
        })

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

        with strategy.scope():
            constant_a = tf.constant(10)
            constant_b = tf.constant(32)

        with tf.device("/job:worker/task:0"):
            op = constant_a + constant_b

        with strategy.scope():
            result = op.numpy()
            print("Result is: ")
            print(result)
            c.stop()

if __name__ == '__main__':
    main()
