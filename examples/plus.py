from __future__ import print_function

import tensorflow as tf
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
        with tf.device('/job:ps/task:0'):
            a = tf.constant(10)

        with tf.device('/job:ps/task:1'):
            b = tf.constant(32)

        with tf.device("/job:worker/task:1"):
            op = a + b

        with tf.compat.v1.Session(c.targets['/job:worker/task:0']) as sess:
            print(sess.run(op))    
     

if __name__ == '__main__':
    main()
