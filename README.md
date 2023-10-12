# Tensorflow V2 for Mesos

With these python module it's possible to run Tensorflow scripts against a Mesos cluster. For 
detailed examples, please have a look into the examples directory.

## Requirements

- Apache Mesos minimum 1.6.x

## How to install

To install tensorflow-mesos and all required packages, execute the following command.

```bash

pip install tfmesos2

```

## How to use?

Tensorflow for Mesos need some environment variables to know how and which Mesos it should use.

```bash

export MESOS_SSL=true
export MESOS_MASTER=localhost:5050
export MESOS_USERNAME=<MESOS_PRINCIPAL>
export MESOS_PASSWORD=<MESOS_SECRET>

python examples/plus.py

```
