import cv2
import numpy as np
import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse


ROOT_DIRECTORY = str(Path(__file__).absolute().parent.parent)


class Preprocessing(object):
    """docstring"""
    def __init__(self, id: str):
        self.id = id

        self.max_frame_num = 100000
        self.vid_file = os.path.join('cloud', 'video', id+'.mp4')

        self.id_dir = os.path.join('cloud', id)
        Path(ROOT_DIRECTORY + '/' + self.id_dir).mkdir(parents=True, exist_ok=True)

        self.ori_imgs_dir = os.path.join('cloud', id, 'ori_imgs')
        Path(ROOT_DIRECTORY + '/' + self.ori_imgs_dir).mkdir(parents=True, exist_ok=True)

        self.valid_img_num = None
        self.h = None
        self.w = None

    def height_weight(self):
        max_frame_num = 100000
        valid_img_ids = []
        for i in range(max_frame_num):
            if os.path.isfile(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, str(i) + '.lms')):
                valid_img_ids.append(i)
        self.valid_img_num = len(valid_img_ids)
        tmp_img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, str(valid_img_ids[0]) + '.jpg'))
        self.h, self.w = tmp_img.shape[0], tmp_img.shape[1]

    def euler2rot(self, euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                         device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                           device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

    def step_0(self):
        print('--- Step0: extract deepspeech feature ---')
        wav_file = os.path.join(self.id_dir, 'aud.wav')
        extract_wav_cmd = 'ffmpeg -i ' + ROOT_DIRECTORY + '/' + self.vid_file + ' -f wav -ar 16000 ' + ROOT_DIRECTORY + '/' + wav_file
        os.system(extract_wav_cmd)
        print(self.id_dir)
        extract_ds_cmd = f'python {ROOT_DIRECTORY}/AD-NeRF/data_util/deepspeech_features/extract_ds_features.py --input=' + ROOT_DIRECTORY + '/' + self.id_dir
        os.system(extract_ds_cmd)
        #exit()

    def step_1(self):
        print('--- Step1: extract images from vids ---')
        print()
        cap = cv2.VideoCapture(ROOT_DIRECTORY + '/' + self.vid_file)
        frame_num = 0
        while (True):
            _, frame = cap.read()
            if frame is None:
                break
            cv2.imwrite(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, str(frame_num) + '.jpg'), frame)
            frame_num = frame_num + 1
        cap.release()
        #exit()

    def step_2(self):
        print('--- Step 2: detect landmarks ---')
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False)
        for image_path in os.listdir(ROOT_DIRECTORY + '/' + self.ori_imgs_dir):
            if image_path.endswith('.jpg'):
                input = io.imread(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, image_path))[:, :, :3]
                preds = fa.get_landmarks(input)
                if len(preds) > 0:
                    lands = preds[0].reshape(-1, 2)[:, :2]
                    np.savetxt(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, image_path[:-3] + 'lms'), lands, '%f')
        self.height_weight()

    def step_3(self):
        print('--- Step 3: face parsing ---')
        face_parsing_cmd = f'python {ROOT_DIRECTORY}/AD-NeRF/data_util/face_parsing/test.py --respath={ROOT_DIRECTORY}/cloud/' + \
                           self.id + f'/parsing --imgpath={ROOT_DIRECTORY}/cloud/' + self.id + '/ori_imgs'
        os.system(face_parsing_cmd)

    def step_6(self):
        print('--- Estimate Head Pose ---')
        est_pose_cmd = f'python {ROOT_DIRECTORY}/AD-NeRF/data_util/face_tracking/face_tracker.py --idname=' + \
                       self.id + ' --img_h=' + str(self.h) + ' --img_w=' + str(self.w) + \
                       ' --frame_num=' + str(self.max_frame_num)
        os.system(est_pose_cmd)
        #exit()


if __name__ == "__main__":
    preproc = Preprocessing(id='premier')
    # preproc.step_0()
    # preproc.step_1()
    # preproc.step_2()
    preproc.height_weight()
    preproc.step_6()
    # print(Path(__file__).absolute().parent.parent)

