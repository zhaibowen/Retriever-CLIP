import os
import re
import cv2
import time
import copy
import json
import torch
import random
import pickle
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizerFast
from config import CLIPConfig_medium
from torchvision.transforms.functional import rotate, resize, adjust_brightness, adjust_saturation, adjust_hue, adjust_contrast, InterpolationMode

class GPTDataset(Dataset):
    def __init__(self, token_dump_path, transform=None):
        self.transform = transform
        self.images, self.tokens = [], []
        self.image_list, self.token_list, self.len_list = [], [], []
        self.len = 0
        for tp, snum in token_dump_path:
            images, tokens = pickle.load(open(tp, 'rb'))
            self.image_list.append(images)
            self.token_list.append(tokens)
            self.len_list.append(snum)
            self.len += snum if snum else len(images)
    
    def __len__(self):
        return self.len
    
    def prepare_sample(self, index):
        index %= 8
        self.images, self.tokens = [], []
        for images, tokens, slen in zip(self.image_list, self.token_list, self.len_list):
            if slen:
                self.images += images[index * slen : (index + 1) * slen]
                self.tokens += tokens[index * slen : (index + 1) * slen]
            else:
                self.images += images
                self.tokens += tokens

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.transform:
            img = self.transform(img)
        token_id = np.random.randint(len(self.tokens[index]))
        return img/255., self.tokens[index][token_id]

class ImageTransformer(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def get_size(self, scale_h):
        t_top = s_top = 0
        t_bottom = s_bottom = self.img_size
        if scale_h > self.img_size:
            t_top = 0
            t_bottom = self.img_size
            s_top = np.random.randint(0, scale_h - self.img_size)
            s_bottom = s_top + self.img_size
        elif scale_h < self.img_size:
            t_top = np.random.randint(0, self.img_size - scale_h)
            t_bottom = t_top + scale_h
            s_top = 0
            s_bottom = scale_h
        return t_top, t_bottom, s_top, s_bottom

    def __call__(self, img):
        _, H, W = img.shape
        # cv2.imshow('x', img.permute(1,2,0).numpy())
        scale_ratio = np.random.uniform(0.8, 1.2) * self.img_size / max(H, W)
        scale_h = np.random.uniform(0.9, 1.1) * H
        scale_w = np.random.uniform(0.9, 1.1) * W
        scale_h2 = int(scale_h * scale_ratio)
        scale_w2 = int(scale_w * scale_ratio)

        angle = np.random.uniform(-5.0, 5.0)
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        saturation = np.random.uniform(0.9, 1.1)
        hue = np.random.uniform(-0.02, 0.02)
        
        img = resize(img, [scale_h2, scale_w2], InterpolationMode.NEAREST)
        img_new = torch.zeros((3, self.img_size, self.img_size), dtype=img.dtype)
        t_top, t_bottom, s_top, s_bottom = self.get_size(scale_h2)
        t_left, t_right, s_left, s_right = self.get_size(scale_w2)
        img_new[:, t_top : t_bottom, t_left : t_right] = img[:, s_top : s_bottom, s_left : s_right]
        
        img_new = rotate(img_new, angle)
        img_new = adjust_brightness(img_new, brightness)
        img_new = adjust_contrast(img_new, contrast)
        img_new = adjust_saturation(img_new, saturation)
        img_new = adjust_hue(img_new, hue)
        # cv2.imshow('y', img_new.permute(1,2,0).numpy())
        # cv2.waitKey(0)
        return img_new

class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last: self.total_size += batch_size

        if shuffle: random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch=0):
        random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

class DistRandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % (self.num_replicas * self.batch_size)
        if not drop_last: self.total_size += self.num_replicas * self.batch_size

        if shuffle:
            g = torch.Generator()
            g.manual_seed(-1)
            self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch):
        g = torch.Generator()
        g.manual_seed(epoch)
        self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

def FixCollector(batch):
    images, tokens = zip(*batch)
    images = torch.stack(images)
    tokens = torch.from_numpy(np.array(tokens)).to(torch.long)
    return images, tokens

if __name__ == "__main__":
    model_root = "/home/work/disk/vision/retriever-clip"
    tokenizer_path = "pretrain/tokenizer_v2_600G.json"
    token_dump_path = "checkpoint/tokens.pkl"
    data_root = "/home/work/disk/coco"
    caption_file = "annotations/captions_train2017.json"
    image_dir = "images/train2017"
    config = CLIPConfig_medium()
    load_token = False

    tokenizer_path = os.path.join(model_root, tokenizer_path)
    token_dump_path = os.path.join(model_root, token_dump_path)
    caption_file = os.path.join(data_root, caption_file)
    image_dir = os.path.join(data_root, image_dir)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    dataset = GPTDataset(transform=ImageTransformer(config.img_size))
    if load_token:
        dataset.load_samples(token_dump_path)
    else:
        dataset.tokenize(caption_file, image_dir, tokenizer, config.sequence_length, token_dump_path)
    sampler = RandSampler(dataset, batch_size=4, drop_last=True, shuffle=False)
    dataloader = DataLoader(dataset, num_workers=0, pin_memory=True, collate_fn=FixCollector, batch_sampler=sampler)
    
    for j, data in enumerate(dataloader):
        a = 1