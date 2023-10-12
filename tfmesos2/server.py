import tensorflow as tf
import logging
import requests
import time
import socket
import os
import sys
import json

client_ip = ""
client_username = ""
client_password = ""
task_id = ""
port = ""


def set_port():
    url = f"http://{client_ip}/v0/task/{task_id}/port/{port}"

    auth = (client_username, client_password)

    try:
        response = requests.put(url, auth=auth, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Could not connect to tfmesos2 client: " + str(e))

def set_init():
    url = f"http://{client_ip}/v0/task/{task_id}"

    auth = (client_username, client_password)

    try:
        response = requests.put(url, auth=auth, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Could not connect to tfmesos2 client: " + str(e))

def get_status():
    url = f"http://{client_ip}/v0/status"

    auth = (client_username, client_password)

    try:
        response = requests.get(url, auth=auth, verify=False)
        response.raise_for_status()
        if response.status_code == 200 and response.text == "ok":
            return True
    except requests.exceptions.RequestException as e:
        logger.error("Could not connect to tfmesos2 client: " + str(e))

    
    return False

def get_cluster_def():
    url = f"http://{client_ip}/v0/task/{task_id}/job"

    auth = (client_username, client_password)

    try:
        response = requests.get(url, auth=auth, verify=False)
        response.raise_for_status()
        print(response.text)
        if response.status_code == 200:
            return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error("Could not connect to tfmesos2 client: " + str(e))

    return None


def loop():
    server = None
    while True:
        if server is None:
            if client_ip != "":
                set_port()
            if get_status():
                job_info = get_cluster_def()
                if job_info is not None:
                    cluster_def=tf.train.ClusterSpec(job_info["cluster_def"])

                    job_name=job_info["job_name"]
                    task_index=job_info["task_index"]

                    logger.info(job_info["cluster_def"])
                    logger.info(job_info["job_name"])
                    logger.info(job_info["task_index"])


                    server = tf.distribute.Server(cluster_def, job_name=job_name, task_index=task_index, protocol="grpc", config=tf.compat.v1.ConfigProto(allow_soft_placement=True), start=True)

                    cpu = tf.config.list_logical_devices()
                    gpu = tf.config.list_physical_devices()

                    logger.info(cpu)
                    logger.info(gpu)


                    try:
                        server.join()
                        set_init()
                    except Exception as e:
                        logger.error("Tensorflow error: " + str(e))

        time.sleep(10)  


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()        

    client_username = os.getenv("TFMESOS2_CLIENT_USERNAME")
    client_password = os.getenv("TFMESOS2_CLIENT_PASSWORD")

    task_id, client_ip = sys.argv[1:]

    loop()








