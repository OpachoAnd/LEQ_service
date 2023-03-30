from handlers.preprocessing_video import preprocessing_processes
from handlers.train_head_nerf import train_head
import torch
import torch.multiprocessing as mp
from handlers.transfer_weights_head import transfer_weights
from handlers.train_torso_nerf import train_torso
from pathlib import Path

HOME_DIRECTORY = str(Path(__file__).absolute().parent)


if __name__ == "__main__":
    video_id = "Obama1"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mp.set_start_method(method='spawn', force=True)

    path_config_head = f'{HOME_DIRECTORY}/cloud/{video_id}/HeadNeRF_config.txt'
    path_config_torso = f'{HOME_DIRECTORY}/cloud{video_id}/TorsoNeRF_config.txt'
    path_weight_head = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_head'
    path_weight_torso = f'{HOME_DIRECTORY}/cloud/{video_id}/logs/{video_id}_com'

    print(path_config_head)
    preprocessing_processes(video_id)
    train_head(path_config_head)
    # transfer_weights(path_weight_head, path_weight_torso)
    # train_torso(path_config_head)
