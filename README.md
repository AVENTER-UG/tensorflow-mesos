# Tensorflow V2 for Mesos

[![Docs](https://img.shields.io/static/v1?label=&message=Issues&color=brightgreen)](https://github.com/AVENTER-UG/tensorflow-mesos/issues)
[![Chat](https://img.shields.io/static/v1?label=&message=Chat&color=brightgreen)](https://matrix.to/#/#mesos:matrix.aventer.biz?via=matrix.aventer.biz)

With these python module it's possible to run Tensorflow scripts against a Mesos cluster. For 
detailed examples, please have a look into the examples directory.

## Funding

[![](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=H553XE4QJ9GJ8)

## Issues

To open an issue, please use this place: https://github.com/AVENTER-UG/tensorflow-mesos/issues

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
