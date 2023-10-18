import logging
import time
import socket
import os
import sys
import json
import requests
import tensorflow as tf

CLIENT_IP = ""
CLIENT_USERNAME = ""
CLIENT_PASSWORD = ""
TASK_ID = ""
PORT = ""
LOGGER = None

def set_port():
    url = f"http://{CLIENT_IP}/v0/task/{TASK_ID}/port/{PORT}"

    auth = (CLIENT_USERNAME, CLIENT_PASSWORD)

    try:
        response = requests.put(url, auth=auth, verify=False, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.error("Could not connect to tfmesos2 client: " + str(e))

def set_init():
    url = f"http://{CLIENT_IP}/v0/task/{TASK_ID}"

    auth = (CLIENT_USERNAME, CLIENT_PASSWORD)

    try:
        response = requests.put(url, auth=auth, verify=False, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.error("Could not connect to tfmesos2 client: " + str(e))

def get_status():
    url = f"http://{CLIENT_IP}/v0/status"

    auth = (CLIENT_USERNAME, CLIENT_PASSWORD)

    try:
        response = requests.get(url, auth=auth, verify=False, timeout=120)
        response.raise_for_status()
        if response.status_code == 200 and response.text == "ok":
            return True
    except requests.exceptions.RequestException as e:
        LOGGER.error("Could not connect to tfmesos2 client: " + str(e))

    return False

def get_cluster_def():
    url = f"http://{CLIENT_IP}/v0/task/{TASK_ID}/job"

    auth = (CLIENT_USERNAME, CLIENT_PASSWORD)

    try:
        response = requests.get(url, auth=auth, verify=False, timeout=120)
        response.raise_for_status()
        print(response.text)
        if response.status_code == 200:
            return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        LOGGER.error("Could not connect to tfmesos2 client: " + str(e))

    return None


def loop():
    server = None
    while True:
        LOGGER.info(server)
        if server is None:
            if CLIENT_IP != "":
                set_port()
            if get_status():
                job_info = get_cluster_def()
                if job_info is not None:
                    cluster_def=tf.train.ClusterSpec(job_info["cluster_def"])

                    os.environ["TF_CONFIG"] = json.dumps({
                        "cluster": job_info["cluster_def"]
                    })

                    job_name=job_info["job_name"]
                    task_index=job_info["task_index"]

                    LOGGER.info(job_info["cluster_def"])
                    LOGGER.info(job_info["job_name"])
                    LOGGER.info(job_info["task_index"])

                    server = tf.distribute.Server(cluster_def, job_name=job_name, task_index=task_index, protocol="grpc", config=tf.compat.v1.ConfigProto(allow_soft_placement=True), start=True)

                    cpu = tf.config.list_logical_devices()
                    gpu = tf.config.list_physical_devices()

                    LOGGER.info(cpu)
                    LOGGER.info(gpu)

                    try:
                        server.join()
                        set_init()
                    except Exception as e:
                        LOGGER.error("Tensorflow error: " + str(e))

        time.sleep(10)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    LOGGER = logging
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    PORT = sock.getsockname()[1]
    sock.close()

    CLIENT_USERNAME = os.getenv("TFMESOS2_CLIENT_USERNAME")
    CLIENT_PASSWORD = os.getenv("TFMESOS2_CLIENT_PASSWORD")

    TASK_ID, CLIENT_IP = sys.argv[1:]

    loop()
