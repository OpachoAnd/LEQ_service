FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /usr/src/app
COPY weights weights/

RUN apt-get update && apt-get install -y git
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN git clone https://github.com/LevchenkoJH/GPEN
RUN cd ./GPEN && git pull
RUN cd ./GPEN && git checkout origin/Для_суперкомпьютера_Четвертый_эксперимент_CLIP_Готовим_Inference
RUN pip install --upgrade pip
RUN cd ./GPEN && pip install -r requirements.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

#CMD ["GPEN/demo.py", "--model", "L-GPEN-256-256-300000",  "--in_size", "256", "--out_size", "256", "--indir", "examples/video", "--outdir", "results/outs-BFR"]

#python GPEN/demo.py --model L-GPEN-256-256-300000 --in_size 256 --out_size 256 --indir indir --outdir outdir
# Добавить ссылку на папку с видео
# Загрузить веса на гугл-диск