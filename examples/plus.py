from __future__ import print_function

import json
import os
import tensorflow as tf
from tfmesos2 import cluster


def main():
    jobs_def = [
        {
            "name": "ps",
            "num":1 
        },
        {
            "name": "worker",
            "num":2 
        },
    ]

    client_ip = "192.168.150.6"

    with cluster(jobs_def, client_ip=client_ip) as c:
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": c.cluster_def
        })

        print(os.environ["TF_CONFIG"])

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

        with strategy.scope():
            constant_a = tf.constant(10)
            constant_b = tf.constant(32)

            op = constant_a + constant_b

        result = op.numpy()
        print("Result is: ")
        print(result)
        c.shutdown()

if __name__ == '__main__':
    main()
