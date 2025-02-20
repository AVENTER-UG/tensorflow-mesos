from __future__ import print_function

import json
import os
import tensorflow as tf
from tfmesos2 import cluster

os.environ["MESOS_MASTER"] = <MESOS_MASTER>:5050"
os.environ["MESOS_SSL"] = "true"
os.environ["MESOS_USERNAME"] = "<MESOS_USERNAME>"
os.environ["MESOS_PASSWORD"] = "<MESOS_PASSWORD>"

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

    client_ip = "<DEVELOPERS_IP>"

    with cluster(jobs_def, client_ip=client_ip) as c:
        os.environ["TF_CONFIG"] = json.dumps({"cluster": c.cluster_def})

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

if __name__ == '__main__':
    main()
