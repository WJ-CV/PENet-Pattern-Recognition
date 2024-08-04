import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from thop import profile
from torch import nn
from Encoder import Mnet
from Decoder import Main_Decoder, Aux_decoders
from data import test_dataset
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gt_root = './data/'
root_ptah = './data/'

Encoder_net = nn.DataParallel(Mnet()).cuda()
Main_Decoder = nn.DataParallel(Main_Decoder()).cuda()

Encoder_net.eval()
Main_Decoder.eval()
fps = 0
if __name__ == '__main__':
    for i in ['model']:  #,'MAE'
        test_datasets = ['feture']
        for dataset in test_datasets:
            time_s = time.time()
            E_model_path = os.path.join('./model/RGBT/', 'E_' + str(i) + '.pth')  #Best_AVG_D_epoch        'E_epoch_' + str(i) +         'D_epoch_' + str(i) +
            D_model_path = os.path.join('./model/RGBT/', 'D_' + str(i) + '.pth')    #Best_AVG_E_epoch
            Encoder_net.load_state_dict(torch.load(E_model_path))
            Main_Decoder.load_state_dict(torch.load(D_model_path))

            sal_save_path = os.path.join('./output/', dataset + '/')
            if not os.path.exists(sal_save_path): os.makedirs(sal_save_path)

            test_loader = test_dataset(gt_root, root_ptah, 384)
            nums = test_loader.size
            for i in range(test_loader.size):
                image, gt, t, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                t = t.cuda()
                out1, out2 = Encoder_net(image, t)
                _, _, score1, _ = Main_Decoder(out1, out2)
                res = F.upsample(score1, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                print('save img to: ', sal_save_path + name)
                cv2.imwrite(sal_save_path + name, res * 255)
            time_e = time.time()
            fps += (nums / (time_e - time_s))
            print("FPS:%f" % (nums / (time_e - time_s)))
            print('Test Done!')
        print("Total FPS %f" % fps) # this result include I/O cost

