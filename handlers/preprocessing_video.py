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

        self.parsing_dir = os.path.join(self.id_dir, 'parsing')
        Path(ROOT_DIRECTORY + '/' + self.parsing_dir).mkdir(parents=True, exist_ok=True)

        self.head_imgs_dir = os.path.join('cloud', id, 'head_imgs')
        Path(ROOT_DIRECTORY + '/' + self.head_imgs_dir).mkdir(parents=True, exist_ok=True)

        self.com_imgs_dir = os.path.join('cloud', id, 'com_imgs')
        Path(ROOT_DIRECTORY + '/' + self.com_imgs_dir).mkdir(parents=True, exist_ok=True)

        self.valid_img_num = None
        self.valid_img_ids = None
        self.h = None
        self.w = None

    def height_weight(self):
        max_frame_num = 100000
        self.valid_img_ids = []
        for i in range(max_frame_num):
            if os.path.isfile(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, str(i) + '.lms')):
                self.valid_img_ids.append(i)
        self.valid_img_num = len(self.valid_img_ids)
        tmp_img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir,
                                          str(self.valid_img_ids[0]) + '.jpg'))
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

    def step_6(self):
        print('--- Estimate Head Pose ---')
        est_pose_cmd = f'python {ROOT_DIRECTORY}/AD-NeRF/data_util/face_tracking/face_tracker.py --idname=' + \
                       self.id + ' --img_h=' + str(self.h) + ' --img_w=' + str(self.w) + \
                       ' --frame_num=' + str(self.max_frame_num)
        os.system(est_pose_cmd)
        #exit()

    def step_3(self):
        print('--- Step 3: face parsing ---')
        face_parsing_cmd = f'python {ROOT_DIRECTORY}/AD-NeRF/data_util/face_parsing/test.py --respath={ROOT_DIRECTORY}/cloud/' + \
                           self.id + f'/parsing --imgpath={ROOT_DIRECTORY}/cloud/' + self.id + '/ori_imgs'
        os.system(face_parsing_cmd)

    def step_4(self):
        print('--- Step 4: extract background image ---')
        sel_ids = np.array(self.valid_img_ids)[np.arange(0, self.valid_img_num, 20)]
        all_xys = np.mgrid[0:self.h, 0:self.w].reshape(2, -1).transpose()
        distss = []
        for i in sel_ids:
            parse_img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.id_dir, 'parsing', str(i) + '.png'))
            bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
            fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
            dists, _ = nbrs.kneighbors(all_xys)
            distss.append(dists)
        distss = np.stack(distss)
        print(distss.shape)
        max_dist = np.max(distss, 0)
        max_id = np.argmax(distss, 0)
        bc_pixs = max_dist > 5
        bc_pixs_id = np.nonzero(bc_pixs)
        bc_ids = max_id[bc_pixs]
        imgs = []
        num_pixs = distss.shape[1]
        for i in sel_ids:
            img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.ori_imgs_dir, str(i) + '.jpg'))
            imgs.append(img)
        imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
        bc_img = np.zeros((self.h * self.w, 3), dtype=np.uint8)
        bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
        bc_img = bc_img.reshape(self.h, self.w, 3)
        max_dist = max_dist.reshape(self.h, self.w)
        bc_pixs = max_dist > 5
        bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
        fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        distances, indices = nbrs.kneighbors(bg_xys)
        bg_fg_xys = fg_xys[indices[:, 0]]
        print(fg_xys.shape)
        print(np.max(bg_fg_xys), np.min(bg_fg_xys))
        bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
        cv2.imwrite(os.path.join(ROOT_DIRECTORY + '/' + self.id_dir, 'bc.jpg'), bc_img)

    def step_5(self):
        print('--- Step 5: save training images ---')
        bc_img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.id_dir, 'bc.jpg'))
        for i in self.valid_img_ids:
            parsing_img = cv2.imread(os.path.join(ROOT_DIRECTORY + '/' + self.parsing_dir, str(i) + '.png'))
            head_part = (parsing_img[:, :, 0] == 255) & (parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0)
            bc_part = (parsing_img[:, :, 0] == 255) & (parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
            img = cv2.imread(ROOT_DIRECTORY + '/' + os.path.join(self.ori_imgs_dir, str(i) + '.jpg'))
            img[bc_part] = bc_img[bc_part]
            cv2.imwrite(os.path.join(ROOT_DIRECTORY + '/' + self.com_imgs_dir, str(i) + '.jpg'), img)
            img[~head_part] = bc_img[~head_part]
            cv2.imwrite(os.path.join(ROOT_DIRECTORY + '/' + self.head_imgs_dir, str(i) + '.jpg'), img)

    def step_7(self):
        print('--- Step 7: Save Transform Param ---')
        params_dict = torch.load(os.path.join(ROOT_DIRECTORY + '/' + self.id_dir, 'track_params.pt'))
        focal_len = params_dict['focal']
        euler_angle = params_dict['euler']
        trans = params_dict['trans'] / 10.0
        valid_num = euler_angle.shape[0]
        train_val_split = int(valid_num * 10 / 11)
        train_ids = torch.arange(0, train_val_split)
        val_ids = torch.arange(train_val_split, valid_num)
        rot = self.euler2rot(euler_angle)
        rot_inv = rot.permute(0, 2, 1)
        trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
        pose = torch.eye(4, dtype=torch.float32)
        save_ids = ['train', 'val']
        train_val_ids = [train_ids, val_ids]
        mean_z = -float(torch.mean(trans[:, 2]).item())
        for i in range(2):
            transform_dict = dict()
            transform_dict['focal_len'] = float(focal_len[0])
            transform_dict['cx'] = float(self.w / 2.0)
            transform_dict['cy'] = float(self.h / 2.0)
            transform_dict['frames'] = []
            ids = train_val_ids[i]
            save_id = save_ids[i]
            for i in ids:
                i = i.item()
                frame_dict = dict()
                frame_dict['img_id'] = int(self.valid_img_ids[i])
                frame_dict['aud_id'] = int(self.valid_img_ids[i])
                pose[:3, :3] = rot_inv[i]
                pose[:3, 3] = trans_inv[i, :, 0]
                frame_dict['transform_matrix'] = pose.numpy().tolist()
                lms = np.loadtxt(os.path.join(ROOT_DIRECTORY, self.ori_imgs_dir, str(self.valid_img_ids[i]) + '.lms'))
                min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
                cx = int((min_x + max_x) / 2.0)
                cy = int(lms[27, 1])
                h_w = int((max_x - cx) * 1.5)
                h_h = int((lms[8, 1] - cy) * 1.15)
                rect_x = cx - h_w
                rect_y = cy - h_h
                if rect_x < 0:
                    rect_x = 0
                if rect_y < 0:
                    rect_y = 0
                rect_w = min(self.w - 1 - rect_x, 2 * h_w)
                rect_h = min(self.h - 1 - rect_y, 2 * h_h)
                rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
                frame_dict['face_rect'] = rect.tolist()
                transform_dict['frames'].append(frame_dict)
            with open(os.path.join(ROOT_DIRECTORY, self.id_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
                json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        testskip = int(val_ids.shape[0] / 7)

        HeadNeRF_config_file = os.path.join(ROOT_DIRECTORY, self.id_dir, 'HeadNeRF_config.txt')
        with open(HeadNeRF_config_file, 'w') as file:
            file.write('expname = ' + self.id + '_head\n')
            file.write('datadir = ' + os.path.join(dir_path, 'cloud', self.id) + '\n')
            file.write('basedir = ' + os.path.join(dir_path, 'cloud', self.id, 'logs') + '\n')
            file.write('near = ' + str(mean_z - 0.2) + '\n')
            file.write('far = ' + str(mean_z + 0.4) + '\n')
            file.write('testskip = ' + str(testskip) + '\n')
        Path(os.path.join(dir_path, 'cloud', self.id, 'logs', self.id + '_head')
             ).mkdir(parents=True, exist_ok=True)

        ComNeRF_config_file = os.path.join(ROOT_DIRECTORY, self.id_dir, 'TorsoNeRF_config.txt')
        with open(ComNeRF_config_file, 'w') as file:
            file.write('expname = ' + self.id + '_com\n')
            file.write('datadir = ' + os.path.join(dir_path, 'cloud', self.id) + '\n')
            file.write('basedir = ' + os.path.join(dir_path, 'cloud', self.id, 'logs') + '\n')
            file.write('near = ' + str(mean_z - 0.2) + '\n')
            file.write('far = ' + str(mean_z + 0.4) + '\n')
            file.write('testskip = ' + str(testskip) + '\n')
        Path(os.path.join(dir_path, 'cloud', self.id, 'logs', self.id + '_com')).mkdir(parents=True, exist_ok=True)

        ComNeRFTest_config_file = os.path.join(ROOT_DIRECTORY, self.id_dir, 'TorsoNeRFTest_config.txt')
        with open(ComNeRFTest_config_file, 'w') as file:
            file.write('expname = ' + self.id + '_com\n')
            file.write('datadir = ' + os.path.join(dir_path, 'cloud', self.id) + '\n')
            file.write('basedir = ' + os.path.join(dir_path, 'cloud', self.id, 'logs') + '\n')
            file.write('near = ' + str(mean_z - 0.2) + '\n')
            file.write('far = ' + str(mean_z + 0.4) + '\n')
            file.write('with_test = ' + str(1) + '\n')
            file.write('test_pose_file = transforms_val.json' + '\n')

        print(self.id + ' data processed done!')

if __name__ == "__main__":
    preproc = Preprocessing(id='premier')
    # preproc.step_0()
    # preproc.step_1()
    # preproc.step_2()
    preproc.height_weight()
    # preproc.step_6()
    # preproc.step_3()
    # preproc.step_4()
    # preproc.step_5()
    preproc.step_7()
    # print(Path(__file__).absolute().parent.parent)

