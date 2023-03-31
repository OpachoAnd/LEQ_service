import os


def extract(home_directory, video_id):
    # Извлечение признаков из аудио файла
    extract_ds_cmd = f'python {home_directory}/ad_nerf/data_util/deepspeech_features/extract_ds_features.py ' \
                     f'--input=' + f'{home_directory}/cloud/video/{video_id}.wav ' \
                                   f'--output=' + f'{home_directory}/cloud/{video_id}/target.npy'
    os.system(extract_ds_cmd)
