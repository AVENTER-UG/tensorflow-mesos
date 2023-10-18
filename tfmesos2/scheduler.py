import os
import json
import math
import threading
import logging
import uuid
import sys
import time
from queue import Queue
import urllib
import urllib3
import requests
from flask import Flask, Response
from addict import Dict
from six import iteritems
from avmesos.client import MesosClient
from waitress import serve

app = Flask(__name__)
TH = None

class Job(object):

    def __init__(self, name, num, cpus=1.0, mem=1024.0,
                 gpus=0, start=0):
        self.name = name
        self.num = num
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.start = start

class Task(object):

    def __init__(self, mesos_task_id, job_name, task_index,
                 cpus=1.0, mem=1024.0, gpus=0, volumes=None, env=None, logger=None):
        self.mesos_task_id = mesos_task_id
        self.job_name = job_name
        self.task_index = task_index

        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.volumes = volumes
        self.env = env
        self.offered = False

        self.addr = None
        self.port = None
        self.connection = None
        self.initialized = False
        self.state = ""
        self.logger = logger

        if volumes is None:
            self.volumes = {}
        if env is None:
            self.env = {}

    def to_task_info(self, offer, master_addr, gpu_uuids=None,
                     gpu_resource_type=None, containerizer_type='DOCKER',
                     force_pull_image=False):
        ti = Dict()
        ti.task_id.value = str(self.mesos_task_id)
        ti.agent_id.value = offer["agent_id"]["value"]
        ti.name = f'/job:{self.job_name}/task:{self.task_index}'
        ti.resources = resources = []

        cpus = Dict()
        resources.append(cpus)
        cpus.name = 'cpus'
        cpus.type = 'SCALAR'
        cpus.scalar.value = self.cpus

        mem = Dict()
        resources.append(mem)
        mem.name = 'mem'
        mem.type = 'SCALAR'
        mem.scalar.value = self.mem

        image = os.getenv("DOCKER_IMAGE", "avhost/tensorflow-mesos:latest")

        if image is not None:
            if containerizer_type == "DOCKER":
                ti.container.type = "DOCKER"
                ti.container.docker.image = image
                ti.container.docker.force_pull_image = force_pull_image
                ti.container.docker.network = "HOST"

                ti.container.docker.parameters = parameters = []
                p = Dict()
                p.key = 'memory-swap'
                p.value = '-1'
                parameters.append(p)

                if self.gpus and gpu_uuids:
                    hostname = offer.hostname
                    g_uuids = urllib.parse.quote(' '.join(gpu_uuids))
                    url = f'http://{hostname}:3476/docker/cli?dev={g_uuids}'

                    try:
                        docker_args = urllib.request.urlopen(url).read()
                        for arg in docker_args.split():
                            k, v = arg.split('=')
                            assert k.startswith('--')
                            k = k[2:]
                            p = Dict()
                            parameters.append(p)
                            p.key = k
                            p.value = v
                    # pylint: disable=W0718
                    except Exception:
                        self.logger.exception(
                            'fail to determine remote device parameter,'
                            ' disable gpu resources'
                        )
                        gpu_uuids = []
            else:
                assert False, f"Unsupported containerizer: {containerizer_type}"

            ti.container.volumes = volumes = []

            for src, dst in iteritems(self.volumes):
                v = Dict()
                volumes.append(v)
                v.container_path = dst
                v.host_path = src
                v.mode = 'RW'

        if self.gpus and gpu_uuids and gpu_resource_type is not None:
            if gpu_resource_type == 'SET':
                gpus = Dict()
                resources.append(gpus)
                gpus.name = 'gpus'
                gpus.type = 'SET'
                gpus.set.item = gpu_uuids
            else:
                gpus = Dict()
                resources.append(gpus)
                gpus.name = 'gpus'
                gpus.type = 'SCALAR'
                gpus.scalar.value = len(gpu_uuids)

        ti.command.shell = True
        cmd = [
            '/usr/bin/python3', '-m', 'tfmesos2.server',
            str(self.mesos_task_id), master_addr
        ]
        ti.command.value = ' '.join(cmd)
        ti.command.environment.variables = variables = [
            Dict(name=name, value=value)
            for name, value in self.env.items()
            if name != 'PYTHONPATH'
        ]
        env = Dict()
        variables.append(env)
        env.name = 'PYTHONPATH'
        env.value = ':'.join(sys.path)
        return ti


class TensorflowMesos(threading.Thread):
    MAX_FAILURE_COUNT = 3

    def run(self):
        self.client.register()

    def __init__(self, task_spec, volumes=None, env=None, loglevel=logging.INFO):
        urllib3.disable_warnings()
        logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s: %(message)s')

        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.stop = False

        self.logger = logging
        self.driver = None
        self.task_queue = Queue()
        self.tasks = {}

        if volumes is None:
            self.volumes = {}
        if env is None:
            self.env = {}

        for job in task_spec:
            for task_index in range(job.start, job.num):
                mesos_task_id = str(uuid.uuid4())
                task = Task(
                    mesos_task_id,
                    job.name,
                    task_index,
                    cpus=job.cpus,
                    mem=job.mem,
                    gpus=job.gpus,
                    volumes=volumes,
                    env=env,
                    logger=self.logger
                )
                self.tasks[mesos_task_id] = task
                self.task_queue.put(task)

        self.framework_name = "tf"
        self.framework_id = None
        self.framework_role = os.getenv("MESOS_FRAMEWORK_ROLE", "tensorflow")

        self.master = os.getenv("MESOS_MASTER", "localhost:5050")
        master_urls = "http://" + self.master
        if os.getenv("MESOS_SSL", "False").lower() == "true":
            master_urls = "https://" + self.master

        self.client = MesosClient(
            mesos_urls=master_urls.split(","),
            frameworkName=self.framework_name,
            frameworkId=None,
        )

        self.logger.info(
            "MesosFramework master : %s, name : %s, id : %s",
            self.master,
            self.framework_name,
            self.framework_id,
        )

        self.client = MesosClient(mesos_urls=master_urls.split(','))
        self.client.principal = os.getenv("MESOS_USERNAME")
        self.client.secret = os.getenv("MESOS_PASSWORD")
        self.client.set_role(self.framework_role)

        self.client.on(MesosClient.SUBSCRIBED, self.subscribed)
        self.client.on(MesosClient.UPDATE, self.status_update)
        self.client.on(MesosClient.OFFERS, self.offer_received)


    def shutdown(self):
        """
        stop the framework

        """
        self.logger.info("Cluster teardown")
        self.stop_event.set()
        self.client.stop = True
        self.client.tearDown()
        self.stop = True
        self.join()

    def subscribed(self, driver):
        """
        Subscribe to Mesos Master

        """
        self.driver = driver


    def status_update(self, update):
        """Update the Status of the Tasks. Based by Mesos Events."""
        task_id = update["status"]["task_id"]["value"]
        task_state = update["status"]["state"]


        self.logger.info("Task %s is in state %s", task_id, task_state)
        self.tasks[task_id].state = task_state

        if task_state == "TASK_RUNNING":
            network_infos = update["status"]["container_status"].get("network_infos")
            if len(network_infos) > 0:
                if len(network_infos[0].get("ip_addresses")) > 0:
                    ip_address = network_infos[0]["ip_addresses"][0].get("ip_address")
                    self.tasks[task_id].addr = ip_address


        if task_state == "TASK_FINISHED":
            self.tasks[task_id] = None

        if task_state in ("TASK_LOST", "TASK_KILLED", "TASK_FAILED"):
            self.task_queue.put(self.tasks[task_id])


    def offer_received(self, offers):
        """If we got a offer, run a queued task"""
        offer_options = {
            "Filters": {
                "RefuseSeconds": 120.0
            }
        }
        len_offers = len(offers)
        if not self.task_queue.empty():
            for index in range(len_offers):
                offer = offers[index]
                if not self.run_job(offer):
                    offertmp = offer.get_offer()
                    self.logger.info("Declined Offer: %s", offertmp["id"]["value"])
                    offer.decline(options=offer_options)
        else:
            for index in range(len_offers):
                offer = offers[index]
                offertmp = offer.get_offer()
                self.logger.info("Declined Offer: %s", offertmp["id"]["value"])
                offer.decline(options=offer_options)

    # pylint: disable=too-many-branches
    def run_job(self, mesos_offer):
        """Start a queued Airflow task in Mesos"""
        offer = mesos_offer.get_offer()
        tasks = []
        option = {}
        offer_cpus = 0.1
        offer_mem = 256.0
        offer_gpus = []
        gpu_resource_type = None
        force_pull = "false"
        container_type = "DOCKER"

        if not self.task_queue.empty():
            task = self.task_queue.get()

            # get CPU, mem and gpu from offer
            for resource in offer["resources"]:
                if resource["name"] == "cpus":
                    offer_cpus = resource["scalar"]["value"]
                elif resource["name"] == "mem":
                    offer_mem = resource["scalar"]["value"]
                elif resource["name"] == "gpus":
                    if resource["type"] == "SET":
                        offer_gpus = resource.set.item
                    else:
                        offer_gpus = list(range(int(resource["scalar"]["value"])))

                    gpu_resource_type = resource["type"]

            gpus = int(math.ceil(task.gpus))
            gpu_uuids = offer_gpus[:gpus]
            offer_gpus = offer_gpus[gpus:]

            self.logger.debug("Received offer %s with cpus: %f and mem: %f for task %s", offer["id"]["value"], offer_cpus, offer_mem, task.mesos_task_id)

            # if the resources does not match, add the task again
            if float(offer_cpus) < float(task.cpus):
                self.logger.info("Offered CPU's for task %s are not enough: got: %f need: %f - %s", task.mesos_task_id, offer_cpus, task.cpus, offer["id"]["value"])
                self.task_queue.put(task)
                return False
            if float(offer_mem) < float(task.mem):
                self.logger.info("Offered MEM's for task %s are not enough: got: %f need: %f - %s", task.mesos_task_id, offer_mem, task.mem, offer["id"]["value"])
                self.task_queue.put(task)
                return False

            self.logger.info("Launching task %s using offer %s", task.mesos_task_id, offer["id"]["value"])


            task = task.to_task_info(
                   offer, "192.168.150.6:11000", gpu_uuids=gpu_uuids,
                   gpu_resource_type=gpu_resource_type,
                   containerizer_type=container_type,
                   force_pull_image=force_pull
            )

            tasks.append(task)
        if len(tasks) > 0:
            mesos_offer.accept(tasks, option)
            return True
        else:
            mesos_offer.decline()
            return False

    def get_task_info(self, task_id):
        url = f"https://{self.master}/tasks/?task_id={task_id}&framework_id={self.client.frameworkId}"
        headers = {
            "Content-Type": "application/json"
        }
        auth = (self.client.principal, self.client.secret)

        try:
            response = requests.post(url, headers=headers, auth=auth, verify=False, timeout=120)
            response.raise_for_status()
            task = response.json()
            return task
        except requests.exceptions.RequestException as e:
            logging.error("Could not connect to mesos-master: %s", str(e))
            return {}

    @property
    def targets(self):
        targets = {}
        for task in iteritems(self.tasks):
            target_name = f'/job:{task.job_name}/task:{task.task_index}'
            grpc_addr = f'grpc://{task.addr}:{task.port}'
            targets[target_name] = grpc_addr
        return targets

    def wait_until_ready(self):
        tasks = sorted(self.tasks.values(), key=lambda task: task.task_index)

        while not all(task.initialized for task in tasks):
            self.logger.info("Cluster not ready")
            time.sleep(10)
        time.sleep(10)
        self.logger.info("Cluster ready")
        self.logger.info("Suppress Mesos Framework")
        self.driver.suppress()

    @property
    def cluster_def(self):
        cluster_def = {}
        tasks = sorted(self.tasks.values(), key=lambda task: task.task_index)

        for task in tasks:
            if task.addr is not None and task.port is not None:
                cluster_def.setdefault(task.job_name, []).append(task.addr+":"+task.port)

        return cluster_def

    @property
    def get_tasks(self):
        return self.tasks

class API(threading.Thread):
    def __init__(self, tasks, port=11000):
        threading.Thread.__init__(self)
        self.tasks = tasks
        self.port = port

    def run(self):
        app.add_url_rule(
            "/v0/task/<task_id>/job",
            "task/<task_id>/job",
            self.get_task_job,
            methods=["GET"],
        )

        app.add_url_rule(
            "/v0/task/<task_id>/port/<port>",
            "task/<task_id>/port/<port>",
            self.set_task_port,
            methods=["PUT"],
        )

        app.add_url_rule(
            "/v0/status",
            "status",
            self.get_status,
            methods=["GET"],
        )

        app.add_url_rule(
            "/v0/task/<task_id>",
            "task/<task_id>",
            self.set_task_init,
            methods=["PUT"],
        )

        serve(app, host="0.0.0.0", port=self.port)

    def set_task_port(self, task_id, port):
        if task_id in self.tasks:
            self.tasks[task_id].port = port
            self.tasks[task_id].initialized = True

        return Response(None, status=200, mimetype="application/json")

    def set_task_init(self, task_id):
        if task_id in self.tasks:
            self.tasks[task_id].initialized = True

        return Response(None, status=200, mimetype="application/json")

    def get_status(self):
        res = "nok"
        tasks = sorted(self.tasks.values(), key=lambda task: task.task_index)

        for task in tasks:
            if task.state == "TASK_RUNNING":
                res = "ok"
            if task.state != "TASK_RUNNING":
                return Response(res, status=204, mimetype="application/json")

        return Response(res, status=200, mimetype="application/json")

    def get_task_job(self, task_id):
        cluster_def = {}

        tasks = sorted(self.tasks.values(), key=lambda task: task.task_index)

        for task in tasks:
            if task.addr is not None and task.port is not None:
                cluster_def.setdefault(task.job_name, []).append(task.addr+":"+task.port)

        if task_id in self.tasks and self.tasks[task_id] is not None:
            data = {
                'job_name': self.tasks[task_id].job_name,
                'task_index': self.tasks[task_id].task_index,
                'cpus':self.tasks[task_id].cpus,
                'mem': self.tasks[task_id].mem,
                'gpus':self.tasks[task_id].gpus,
                'cluster_def': cluster_def
            }

            response = Response(
                json.dumps(data), status=200, mimetype="application/json"
            )
            return response

        response = Response(None, status=200, mimetype="application/json")
        return response
    