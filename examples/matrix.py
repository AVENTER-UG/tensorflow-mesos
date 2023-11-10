from __future__ import print_function

import json
import os
import tensorflow as tf
from tfmesos2 import cluster


def main():
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

    client_ip = "192.168.150.6"

    with cluster(jobs_def, client_ip=client_ip) as c:
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": c.cluster_def
        })

        print(os.environ["TF_CONFIG"])

        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

        with strategy.scope():
            mat_a = tf.constant([[2,3], [1,2], [4,5]])
            mat_b = tf.constant([[6,4,1], [3,7,2]])

            op = tf.matmul(mat_a, mat_b)

        result = op.numpy()
        print("Result is: ")
        print(result)
        c.shutdown()

if __name__ == '__main__':
    main()
