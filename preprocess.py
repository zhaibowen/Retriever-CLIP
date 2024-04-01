import os
import re
import json
import copy
import random
import pickle
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast
from config import CLIPConfig_medium
import urllib.request as ureq
from multiprocessing import Process

def coco_count_token_length():
    data_path = "/home/work/disk/coco/annotations/captions_train2017.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
        captions = list(map(lambda x: x['caption'], data['annotations']))
        captions = list(map(lambda x: re.sub(' +', ' ', x.strip()), captions))

    tokens = tokenizer(captions)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)
    print(tlen.max()) # 67
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5])) # [12. 14. 16. 17. 23. 25.]

def tokenize(caption_file, image_dir, tokenizer, sequence_length, token_dump_path):
    IGNORE_INDEX = -100
    eoc = tokenizer.vocab['</s>']

    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    image_id2file_name = {img['id']:img['file_name'] for img in data['images']}
    images = list(map(lambda x: image_id2file_name[x['image_id']], data['annotations']))
    images = list(map(lambda x: os.path.join(image_dir, x), images))

    captions = list(map(lambda x: x['caption'], data['annotations']))
    captions = list(map(lambda x: re.sub(' +', ' ', x.strip()), captions))
    tokens = tokenizer(captions)['input_ids']
    tokens = list(map(lambda x: x[:sequence_length], tokens)) # truncate
    tokens = list(map(lambda x: [eoc] * (sequence_length - len(x)) + x, tokens)) # padding

    image_maps = {}
    for image, token in zip(images, tokens):
        if image not in image_maps:
            image_maps[image] = [token]
        else:
            image_maps[image].append(token)

    images = list(image_maps.keys())
    tokens = list(image_maps.values())
    pickle.dump((images, tokens), open(token_dump_path, 'wb'))

def coco_train_process():
    data_root = "/home/work/coco"
    caption_file = "annotations/captions_train2017.json"
    image_dir = "images/train2017"
    token_dump_path = "checkpoint/tokens_coco.pkl"

    caption_file = os.path.join(data_root, caption_file)
    image_dir = os.path.join(data_root, image_dir)
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    tokenize(caption_file, image_dir, tokenizer, config.sequence_length, token_dump_path)

def coco_eval_process():
    data_root = "/home/work/coco"
    caption_file = "annotations/captions_val2017.json"
    image_dir = "images/val2017"
    token_dump_path = "checkpoint/tokens_coco_eval.pkl"

    caption_file = os.path.join(data_root, caption_file)
    image_dir = os.path.join(data_root, image_dir)
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    tokenize(caption_file, image_dir, tokenizer, config.sequence_length, token_dump_path)

def url_downloader(start_id, image_urls, interval, tdir):
    for i in range(start_id, len(image_urls), interval):
        url = image_urls[i]
        dir, name = url.split('/')[-2:]
        if not os.path.exists(os.path.join(tdir, dir)):
            os.mkdir(os.path.join(tdir, dir))
        outputFile=os.path.join(tdir, dir, name)
        if not os.path.exists(outputFile):
            try:
                ureq.urlretrieve(url, outputFile)
            except:
                # print(f'{i} not found!')
                pass

def SBU_download_images():
    data_path = "/home/work/disk8T/SBU/0000.parquet"
    data = pd.read_parquet(data_path)
    image_urls = data['image_url'].tolist()
    tdir = "/home/work/disk8T/SBU/images"
    
    for epoch in range(10):
        undownloads = []
        for url in image_urls:
            dir, name = url.split('/')[-2:]
            outputFile=os.path.join(tdir, dir, name)
            if not os.path.exists(outputFile):
                undownloads.append(url)
        image_urls = copy.copy(undownloads)
        print(f"epoch {epoch}, undownload {len(image_urls)}")

        pool_num = 100
        for i in range(pool_num):
            p = Process(target=url_downloader, args=(i, image_urls, pool_num, tdir))
            p.start()

        p.join()

def delete_image(start_id, image_urls, interval, tdir):
    knt = 0
    for i in range(start_id, len(image_urls), interval):
        knt += 1
        if knt % 1000 == 0: print(start_id, knt)
        url = image_urls[i]
        dir, name = url.split('/')[-2:]
        outputFile=os.path.join(tdir, dir, name)
        if os.path.exists(outputFile):
            with open(outputFile, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                print(f'Not complete image {outputFile}')
                os.system(f"rm {outputFile}")

def SBU_detect_premature_image():
    data_path = "/home/work/disk8T/SBU/0000.parquet"
    tdir = "/home/work/disk8T/SBU/images"
    data = pd.read_parquet(data_path)
    image_urls = data['image_url'].tolist()

    pool_num = 10
    for i in range(pool_num):
        p = Process(target=delete_image, args=(i, image_urls, pool_num, tdir))
        p.start()
    p.join()

def SBU_process():
    data_path = "/home/work/disk8T/SBU/0000.parquet"
    # tdir = "/home/work/disk8T/SBU/images"
    tdir = "/home/work/SBU/images"
    token_dump_path = "checkpoint/tokens_SBU.pkl"
    token_dump_path = os.path.join(cur_dir, token_dump_path)

    data = pd.read_parquet(data_path)
    captions = data['caption'].tolist()
    captions = list(map(lambda x: re.sub(' +', ' ', x.strip()), captions))
    image_urls = data['image_url'].tolist()
    image_urls = list(map(lambda x: '/'.join([tdir] + x.split('/')[-2:]), image_urls))

    filtered_data = list(filter(lambda x: os.path.exists(x[0]), zip(image_urls, captions)))
    images, captions = zip(*filtered_data)
    print(len(images))

    tokens = tokenizer(captions)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)

    print(tlen.max()) # 143
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5])) # [15. 23. 27. 30. 39. 43.]

    eoc = tokenizer.vocab['</s>']
    sequence_length = config.sequence_length
    tokens = list(map(lambda x: x[:sequence_length], tokens)) # truncate
    tokens = list(map(lambda x: [eoc] * (sequence_length - len(x)) + x, tokens)) # padding

    image_maps = {}
    for image, token in zip(images, tokens):
        if image not in image_maps:
            image_maps[image] = [token]
        else:
            image_maps[image].append(token)

    images = list(image_maps.keys())
    tokens = list(image_maps.values())
    pickle.dump((images, tokens), open(token_dump_path, 'wb'))

def LlavaPretrain_process():
    data_path = '/home/work/disk/LLaVA-data/LLAVA-Pretrain/blip_laion_cc_sbu_558k.json'
    image_dir = '/home/work/LLAVA-Pretrain/images'

    with open(data_path, 'r') as f:
        data = json.load(f)
        images = list(map(lambda x: os.path.join(image_dir, x['image']), data))
        captions = list(map(lambda x: x['conversations'][1]['value'], data))
        
    print(len(images))

    tokens = tokenizer(captions)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)

    print(tlen.max()) # 70
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5])) # [12. 17. 21. 23. 29. 32.]

    eoc = tokenizer.vocab['</s>']
    sequence_length = config.sequence_length
    tokens = list(map(lambda x: x[:sequence_length], tokens)) # truncate
    tokens = list(map(lambda x: [eoc] * (sequence_length - len(x)) + x, tokens)) # padding

    image_maps = {}
    for image, token in zip(images, tokens):
        if image not in image_maps:
            image_maps[image] = [token]
        else:
            image_maps[image].append(token)

    images = list(image_maps.keys())
    tokens = list(image_maps.values())

    x=list(zip(images, tokens))
    random.shuffle(x)
    images, tokens = zip(*x)

    token_dump_path = "checkpoint/tokens_LLAVA.pkl"
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    pickle.dump((images[:-5000], tokens[:-5000]), open(token_dump_path, 'wb'))

    token_dump_path_eval = "checkpoint/tokens_LLAVA_eval.pkl"
    token_dump_path_eval = os.path.join(cur_dir, token_dump_path_eval)
    pickle.dump((images[-5000:], tokens[-5000:]), open(token_dump_path_eval, 'wb'))

if __name__ == "__main__":
    config = CLIPConfig_medium()
    cur_dir = "/home/work/disk/vision/retriever-clip"
    tokenizer_path = "pretrain/tokenizer_v2_600G.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(cur_dir, tokenizer_path))

    # coco_count_token_length()
    # coco_train_process()
    # coco_eval_process()

    # SBU_download_images()
    # SBU_detect_premature_image()
    # SBU_process()

    LlavaPretrain_process()
