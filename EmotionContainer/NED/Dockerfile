FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install curl -y

# install anaconda
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2022.05-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2022.05-Linux-x86_64.sh -b -p /conda"
RUN /conda/bin/conda init bash
RUN /conda/bin/conda update -n base -c defaults conda -y
ENV PATH /conda/bin:$PATH
RUN conda update --all

COPY ./NED ./app

ENV CUDA_HOME "${CUDA_HOME}/usr/local/cuda-11.0"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN conda env create -f /app/environment.yml

WORKDIR /app

SHELL ["conda", "run", "-n", "NED", "/bin/bash", "-c"]

ENTRYPOINT ["/conda/bin/conda", "run", "-n", "NED", "bash", "run_ned_modules.sh"]

CMD ["happy"]





