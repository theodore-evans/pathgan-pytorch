
FROM docker.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /root/.local/bin:/root/.pyenv/bin:/root/.pyenv/shims:${PATH}

RUN mkdir -p /root/.local/bin \
&& curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

RUN pyenv install 3.8.3 \
&& pyenv global 3.8.3

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH /root/.local/lib:${PYTHONPATH}

RUN MKDIR /root/src

COPY . /root/src/

WORKDIR /root/src