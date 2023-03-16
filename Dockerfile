#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
#FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:10.2-cudnn7-base-ubuntu18.04 (runtime devel)

FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

RUN apt-get update && apt-get install curl -y

# install anaconda
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2022.05-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2022.05-Linux-x86_64.sh -b -p /conda"

#RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
#RUN chmod +x /tmp/Anaconda3-2020.11-Linux-x86_64.sh
#RUN mkdir /root/.conda
#RUN bash -c "/tmp/Anaconda3-2020.11-Linux-x86_64.sh -b -p /conda"

RUN /conda/bin/conda init bash

RUN /conda/bin/conda update -n base -c defaults conda -y

WORKDIR /usr/src/app

COPY ./environment.yml /usr/src/app
RUN /conda/bin/conda env create -f environment.yml

#RUN useradd -m opachoand
#
#RUN chown -R opachoand:opachoand /home/opachoand/
#
#COPY --chown=opachoand . /home/opachoand/app/
#
#USER opachoand





#RUN cd /home/opachoand/app/ && bash -yqq Miniconda3-3.7.sh

#RUN conda env create -f environment.yml

#WORKDIR /home/opachoand/app


#RUN git clone https://github.com/YudongGuo/AD-NeRF.git
#RUN bash Miniconda3-3.7.sh
