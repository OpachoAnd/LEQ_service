FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install curl -y
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get update && apt-get install libgl1-mesa-glx

#nvidia CUB
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
RUN tar xzf 1.10.0.tar.gz
#RUN export CUB_HOME=$PWD/cub-1.10.0
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
ENV CUDA_HOME "${CUDA_HOME}/usr/local/cuda-11.2"

# install environment for ad_nerf
RUN /conda/bin/conda env create -f environment.yml 
#RUN /conda/bin/conda create --name adnerf2 -y
RUN echo conda activate adnerf2 >> /root/.bashrc 
RUN /conda/bin/activate adnerf2 && conda install -c bottler nvidiacub
#RUN /conda/bin/activate adnerf2 && cd pytorch3d && /conda/envs/adnerf2/bin/pip install -e .
RUN /conda/bin/activate adnerf2 && cd ad_nerf/data_util/face_tracking && /conda/envs/adnerf2/bin/python convert_BFM.py
#RUN /conda/bin/conda run --no-capture-output -n adnerf2 pip install -e pytorch3d/.
#ENTRYPOINT ["/conda/envs/adnerf2/bin/pip", "install", "-e", "pytorch3d/." && "/conda/envs/adnerf2/bin/python", "handlers/preprocessing_video.py"]
#CMD /conda/envs/adnerf2/bin/pip install -e pytorch3d/.; /conda/envs/adnerf2/bin/python handlers/preprocessing_video.py

ENTRYPOINT ["/bin/bash", "-c", "/conda/bin/conda run --no-capture-output -n adnerf2 pip install -e pytorch3d/. && /conda/bin/conda run --no-capture-output -n adnerf2 python main.py"]
#CMD ["/conda/envs/adnerf2/bin/python", "handlers/preprocessing_video.py"]

