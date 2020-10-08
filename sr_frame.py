from __future__ import print_function
import argparse

import cv2
import sys, time

import os
import torch
from modules import VAE_SR, VAE_denoise_vali
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
import time
from collections import OrderedDict
import math
from datasets import is_image_file
from image_utils import *
from PIL import Image, ImageOps
from os import listdir
import torch.utils.data as utils
from torch.autograd import Variable
import os

SF = 4

def chop_forward(img):


    img = transform(img).unsqueeze(0)

    testset = utils.TensorDataset(img)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)

    for iteration, batch in enumerate(test_dataloader, 1):
        input = Variable(batch[0]).cuda(gpus_list[0])
        batch_size, channels, img_height, img_width = input.size()

        lowres_patches = patchify_tensor(input, patch_size=opt.patch_size, overlap=opt.stride)

        n_patches = lowres_patches.size(0)
        out_box = []
        with torch.no_grad():
            for p in range(n_patches):
                LR_input = lowres_patches[p:p + 1]
                std_z = torch.from_numpy(np.random.normal(0, 1, (input.shape[0], 512))).float()
                z = Variable(std_z, requires_grad=False).cuda(gpus_list[0])
                Denoise_LR = denoiser(LR_input, z)
                SR = model(Denoise_LR)
                out_box.append(SR)

            out_box = torch.cat(out_box, 0)
            SR = recompose_tensor(out_box, opt.upscale_factor * img_height, opt.upscale_factor * img_width,
                                              overlap=opt.upscale_factor * opt.stride)


            SR = SR.data[0].cpu().permute(1, 2, 0)

    return SR

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=SF, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=64, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=8, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input', type=str, default='Test', help='Location to input images')
parser.add_argument('--model_type', type=str, default='VAE')
parser.add_argument('--output', default='Result', help='Location to save SR results')
parser.add_argument('--model_denoiser', default='models/VAE_denoiser.pth', help='pretrained denoising model')
parser.add_argument('--model_SR', default='models/VAE_SR.pth', help='pretrained SR model')

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)


denoiser = VAE_denoise_vali(input_dim=3, dim=32, feat_size=8, z_dim=512, prior='standard')
model = VAE_SR(input_dim=3, dim=64, scale_factor=opt.upscale_factor)

denoiser = torch.nn.DataParallel(denoiser, device_ids=gpus_list)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
if cuda:
    denoiser = denoiser.cuda(gpus_list[0])
    model = model.cuda(gpus_list[0])


print('===> Loading datasets')

if os.path.exists(opt.model_denoiser):
    # denoiser.load_state_dict(torch.load(opt.model_denoiser, map_location=lambda storage, loc: storage)) 
    pretrained_dict = torch.load(opt.model_denoiser, map_location=lambda storage, loc: storage)
    model_dict = denoiser.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    denoiser.load_state_dict(model_dict)
    print('Pre-trained Denoiser model is loaded.')

if os.path.exists(opt.model_SR):
    model.load_state_dict(torch.load(opt.model_SR, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

def super_res():

    denoiser.eval()
    model.eval()
    
super_res()

def frame_transform_complex(frame):
    with torch.no_grad():
        prediction = chop_forward(frame)
    prediction *= 255.0
    sr_image = prediction.clamp(0, 255)
    
    return np.uint8(sr_image.numpy())
    
def frame_transform_simple(frame, width_new, height_new):
    dim = (width_new, height_new)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
        
    return frame

def vid_read(vid_path):
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    cap= cv2.VideoCapture(vid_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cap.get(cv2.CAP_PROP_FOURCC)
    print('Video Width: {}, Height: {}, FPS: {}, No. Frames: {}, Codec: {}.'.format(width, height, fps, no_frames, codec))

    return vid_name, cap, width, height, fps

def frame_write(cap, out, f_ow, f_nw, out_dir, resize, width_new, height_new, trans_complex=False):
    
    i=0
    time_read_frame_cum = 0
    time_transform_cum = 0
    if trans_complex == True:
        resize=False

    while(cap.isOpened()):
        time_read_frame_start = time.time()
        ret, frame = cap.read()
        if ret == False:
            break
        time_read_frame_end = time.time()
        time_read_frame_cum += time_read_frame_end - time_read_frame_start
        if f_ow != False:
            cv2.imwrite(os.path.join(out_dir, '{}_old.png'.format(i)), frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        time_transform_start = time.time()
        if resize == True:
            transformed_frame = frame_transform_simple(frame, width_new, height_new)
        elif trans_complex == True:
            transformed_frame = frame_transform_complex(frame)
        time_transform_end = time.time()
        time_transform_cum += time_transform_end - time_transform_start

        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)
        out.write(transformed_frame)
        if f_nw != False:
            cv2.imwrite(os.path.join(out_dir, '{}_new.png'.format(i)), transformed_frame)

        if (i%100==0):
            print('Added frame: ', i) 
        i += 1
        
    return time_read_frame_cum, time_transform_cum, i 
    

def vid_transform(vid_path, f_ow, f_nw, width_new, height_new, fps_new, codec_new, trans_complex):
    time_start = time.time()
    vid_name, cap, width, height, fps = vid_read(vid_path)
    
    if width_new == False:
        width_new = width
    else:
        resize = True
    if height_new == False:
        height_new = height
    else:
        resize = True
    if fps_new == False:
        fps_new = fps
    codec_new = tuple(char.upper() for char in codec_new)
        
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, '{}_trans.avi'.format(vid_name)) 

    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(codec_new[0], codec_new[1], codec_new[2], codec_new[3]), fps_new, (width_new, height_new)) #lossy but works
        
    print('Started transforming frame by frame from video.')
    time_read_frame_cum, time_transform_cum, i = frame_write(cap, out, f_ow, f_nw, out_dir, resize, width_new, height_new, trans_complex)
    print('Finished transforming video.')
    cap.release()
    out.release()
    
    return time_start, time_read_frame_cum, time_transform_cum, i

def times(time_start, time_read_frame_cum, time_transform_cum, i):
    
    time_end = time.time()
    time_total = time_end - time_start
    print('Total time {} s. Avg/frame: {} s.'.format(str(time_total), str(time_total/i)))
    time_read_frame_avg = time_read_frame_cum/i
    print('Total time spent reading frames {} s. Avg/frame: {} s.'.format(time_read_frame_cum, time_read_frame_avg))
    time_transform_avg = time_transform_cum/i
    print('Total time spent processing frames {} s. Avg/frame: {} s.'.format(time_transform_cum, time_transform_avg))
    
def main():
    vid_transform('./../../UBFC_og/S13.avi', f_ow=True, f_nw=True, width_new=int(640*SF), height_new=int(480*SF), fps_new=False, codec_new='mjpg', trans_complex=True)
main()