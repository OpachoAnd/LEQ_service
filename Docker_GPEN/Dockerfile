FROM python:3.9

WORKDIR /usr/src/app

# В этой ветке буду все клонировать из репозитория (не используя сабмодуль)
RUN git clone https://github.com/LevchenkoJH/GPEN
RUN cd ./GPEN && git pull
RUN cd ./GPEN && git checkout origin/Для_суперкомпьютера_Четвертый_эксперимент_CLIP_Готовим_Inference
RUN pip install --upgrade pip
RUN cd ./GPEN && pip install -r requirements.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113