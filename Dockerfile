FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
RUN apt-get update && apt-get install --no-install-recommends -y gcc ca-certificates python3-pip && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt 

COPY . /tmp/tfmesos2

RUN cd /tmp/tfmesos2; pip3 install .
RUN rm -rf /tmp/tfmesos2
ENV DOCKER_IMAGE avhost/tensorflow-mesos
