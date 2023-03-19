FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

RUN apt-get update && apt-get install curl -y
RUN apt-get update && apt-get install -y git


# install anaconda
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2022.05-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2022.05-Linux-x86_64.sh -b -p /conda"
RUN /conda/bin/conda init bash
RUN /conda/bin/conda update -n base -c defaults conda -y

# import folders
WORKDIR /usr/src/app
COPY . /usr/src/app

# install environment for AD-NeRF
RUN /conda/bin/conda env create -f environmentTest.yml 
#RUN /conda/bin/conda create --name adnerf2 -y
RUN echo conda activate adnerf2 >> /root/.bashrc 
RUN /conda/bin/activate adnerf2 && cd pytorch3d && /conda/envs/adnerf2/bin/pip install -e.
RUN /conda/bin/activate adnerf2 && cd AD-NeRF/data_util/face_tracking && /conda/envs/adnerf2/bin/python convert_BFM.py







