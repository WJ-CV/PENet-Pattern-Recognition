import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=111, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch_list', type=int, default=[21, 68, 91], help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='1, 3', help='train use gpu')

# RGB-T Datasets   /data/wj/Train/supervised1
parser.add_argument('--sup_gt_root', type=str, default='../Train/supervised/', help='the training GT images root')
parser.add_argument('--unsup_gt_root', type=str, default='../Train/unsupervised/', help='the training GT images root')
parser.add_argument('--root_path', type=str, default='../Train/', help='the training GT images root')

parser.add_argument('--test_gt_root', type=str, default='../Val/', help='the training GT images root')
parser.add_argument('--test_root_path', type=str, default='../Val/', help='the training GT images root')
parser.add_argument('--save_path', type=str, default='./model/', help='the path to save models')
opt = parser.parse_args()