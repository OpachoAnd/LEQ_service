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

HOME_DIRECTORY = str(Path(__file__).absolute().parent)


if __name__ == "__main__":
    video_id = "greta7.3"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mp.set_start_method(method='spawn', force=True)

    path_config_head = f'{HOME_DIRECTORY}/cloud/{video_id}/HeadNeRF_config.txt'
    path_config_torso = f'{HOME_DIRECTORY}/cloud{video_id}/TorsoNeRF_config.txt'
    path_weight_head = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_head'
    path_weight_torso = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_com'
    path_config_inference = f'{HOME_DIRECTORY}/cloud/{video_id}/TorsoNeRFTest_config.txt'
    path_audio = f'{HOME_DIRECTORY}/cloud/{video_id}/aud.npy'
    # preprocessing_processes(video_id)

    # train_head(path_config_head)
    # transfer_weights(path_weight_head, path_weight_torso)
    # train_torso(path_config_head)

    # extract(HOME_DIRECTORY, video_id)
    concatenation(HOME_DIRECTORY, video_id)
    inference(path_config_inference, path_audio)
