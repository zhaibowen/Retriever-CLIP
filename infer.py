import os
import cv2
import json
import torch
import numpy as np
from model import retriever_clip
from config import CLIPConfig_medium
from torch.cuda.amp import autocast
from flask import Flask, request
from transformers import PreTrainedTokenizerFast
from torchvision.transforms.functional import resize, InterpolationMode

app = Flask(__name__)

config = CLIPConfig_medium()
flash = True
arch = retriever_clip
dtype = "bfloat16"
tokenizer_path = "/home/work/disk/vision/retriever-clip/pretrain/tokenizer_v2_600G.json"
model_path = "/home/work/disk/vision/retriever-clip/checkpoint/retriever_clip_medium_resnet50_loss0.208_0.180.pth.tar"

device = torch.device(f'cuda')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
model = arch(device, ptdtype, config, True, model_path, flash=flash)
model.cuda()
model.eval()

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
eoc = tokenizer.vocab['</s>']

@app.route("/infer", methods=["POST", "GET"])
def infer():
    data = request.get_json()
    image_path = data['image_path']
    caption = data['caption']

    tokens = tokenizer(caption)['input_ids']
    tokens = tokens[:config.sequence_length] # truncate
    tokens = [eoc] * (config.sequence_length - len(tokens)) + tokens # padding
    input_ids = torch.from_numpy(np.array([tokens])).to(torch.long)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    _, H, W = img.shape
    scale_ratio = config.img_size / max(H, W)
    H2 = int(H * scale_ratio)
    W2 = int(W * scale_ratio)
    img = resize(img, [H2, W2], InterpolationMode.NEAREST)
    img_new = torch.zeros((3, config.img_size, config.img_size), dtype=img.dtype)
    img_new[:, :H2, :W2] = img
    img = img_new/255.
    images = torch.stack([img])

    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        with autocast(dtype=ptdtype):
            similarity = model(input_ids, images, infer=True)
            similarity = float(similarity)
            print(similarity)

    return json.dumps(similarity)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060)

'''
0 '/home/work/coco/images/val2017/000000179765.jpg' 'A black Honda motorcycle parked in front of a garage.'
2 '/home/work/coco/images/val2017/000000190236.jpg' 'An office cubicle with four different types of computers.'
3 '/home/work/coco/images/val2017/000000331352.jpg' 'A small closed toilet in a cramped space.'
4 '/home/work/coco/images/val2017/000000517069.jpg' 'Two women waiting at a bench next to a street.'
50 '/home/work/coco/images/val2017/000000172330.jpg' 'A grey and white cat watches from between parked cars.'
83 '/home/work/coco/images/val2017/000000289393.jpg' 'Several toy animals - a bull, giraffe, deer and parakeet.'
99 '/home/work/coco/images/val2017/000000338325.jpg' 'A plane flies through the sky at an angle.'
'''