FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install curl -y
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get update && apt-get install libgl1-mesa-glx

#nvidia CUB
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
RUN tar xzf 1.10.0.tar.gz

ENV CUB_HOME=$PWD/cub-1.10.0

# install anaconda
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2022.05-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2022.05-Linux-x86_64.sh -b -p /conda"
RUN /conda/bin/conda init bash
RUN /conda/bin/conda update -n base -c defaults conda -y
ENV PATH /conda/bin:$PATH
RUN conda update --all

# import folders
WORKDIR /usr/src/app
COPY . /usr/src/app
ENV PYTHONPATH /usr/src/app
#EXPOSE 5672


# install environment for WEB_INTERFACE
RUN /conda/bin/conda env create -f environment.yml 
RUN echo conda activate web_interface >> /root/.bashrc 
EXPOSE 5000
ENTRYPOINT ["/bin/bash", "-c", "/conda/bin/conda run --no-capture-output -n web_interface python main_api.py"]


