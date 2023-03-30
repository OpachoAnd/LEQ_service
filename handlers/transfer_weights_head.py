import os
import shutil


def path_max_weights(path_weight_head):
    # Возврат пути к весам максимальной итерации
    weights = []
    for x in os.listdir(path_weight_head):
        if x.endswith("_head.tar"):
            weights.append(x)
    return os.path.join(path_weight_head, max(weights))


def transfer_weights(path_weight_head, path_weight_torso):
    # Перенос весов из head в torso для обучения torso
    path_weight = path_max_weights(path_weight_head)
    shutil.copy(path_weight, path_weight_torso)
