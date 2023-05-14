import json
from pathlib import Path

import torch
import torch.multiprocessing as mp

from handlers.inference_nerf import inference
from handlers.preprocessing_video import preprocessing_processes
from handlers.train_head_nerf import train_head
from handlers.train_torso_nerf import train_torso
from handlers.transfer_weights_head import transfer_weights
from handlers.extract_audio import extract
from handlers.transforms_concat import concatenation

import pika

HOME_DIRECTORY = str(Path(__file__).absolute().parent)
RMQ_HOST = 'mq'
RMQ_PORT = 5672
RMQ_USER = 'guest'
RMQ_PASSWORD = 'guest'


def handling_video_callback(ch, method, properties, body):
    print('V OBRABOTKE')
    data = json.loads(body)
    print(data, flush=True)
    video_id = data['video']  # body
    audio_id = data['audio']
    emotion = data['emotion']
    e_mail = data['e_mail']

    path_config_head = f'{HOME_DIRECTORY}/cloud/{video_id}/HeadNeRF_config.txt'
    path_config_torso = f'{HOME_DIRECTORY}/cloud{video_id}/TorsoNeRF_config.txt'
    path_weight_head = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_head'
    path_weight_torso = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_com'
    path_config_inference = f'{HOME_DIRECTORY}/cloud/{video_id}/TorsoNeRFTest_config.txt'
    path_audio = f'{HOME_DIRECTORY}/cloud/{audio_id}/target.npy  # aud.npy'

    preprocessing_processes(video_id)
    train_head(path_config_head)
    transfer_weights(path_weight_head, path_weight_torso)
    train_torso(path_config_torso)  # был path_config_head
    extract(HOME_DIRECTORY, video_id)  # ToDo проверить с новой аудио дорожкой
    concatenation(HOME_DIRECTORY, video_id)
    inference(path_config_inference, path_audio)


if __name__ == "__main__":
    video_id = "greta7.3"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mp.set_start_method(method='spawn', force=True)

    try:
        print("ZASHLI V AD_NERF")
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RMQ_HOST,
                                                                       port=RMQ_PORT))

        channel = connection.channel()
        channel.queue_declare(queue='ad_nerf')
        channel.basic_consume(queue='ad_nerf',
                              on_message_callback=handling_video_callback,
                              auto_ack=True)
        channel.start_consuming()
        print('ok_connection', flush=True)

    except:
        print('error_connection', flush=True)
