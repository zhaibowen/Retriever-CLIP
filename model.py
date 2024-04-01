import math
import copy
import torch
import numpy as np
import torch.nn as nn
from resnet import ResNet, BasicBlock
from resnet_v15 import ResNetV15, Bottleneck
from retriever import Retriever

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, output_size, flash=True):
        super(CrossAttention, self).__init__()
        self.flash=flash
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, query, key, value):
        bsz, q_len, _ = query.size()
        _, k_len, _ = key.size()

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.flash:
            output = nn.functional.scaled_dot_product_attention(query, key, value)
        else:
            att = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert att.size() == (bsz, self.num_heads, q_len, k_len), "Attention weights shape error"
            att = nn.functional.softmax(att, dim=-1)
            output = torch.matmul(att, value)

        output = output.transpose(1, 2).contiguous()
        assert output.size() == (bsz, q_len, self.num_heads, self.head_dim), "Attention output shape error"
        output = output.reshape(bsz, q_len, self.hidden_size)

        output = self.o_proj(output)
        return output

class CLIP(nn.Module):
    def __init__(self, device, ptdtype, config, flash):
        super(CLIP, self).__init__()
        self.hidden_size = config.hidden_size
        self.norm_scale = torch.log(torch.tensor(config.batch_size))
        if config.vision == "resnet18":
            self.image_encoder = ResNet(BasicBlock, config.res_layers, config.res_channels)
        elif config.vision == "resnet50_v15":
            self.image_encoder = ResNetV15(Bottleneck, config.res_layers, config.res_channels)
        self.positional_embedding = nn.Parameter(torch.randn(config.spacial_dim ** 2 + 1, config.img_hidden_size) / config.img_hidden_size ** 0.5)
        self.img_attn = CrossAttention(config.img_hidden_size, config.img_num_head, config.hidden_size, flash)

        self.freeze_layers = config.num_layers - 1
        self.text_encoder = Retriever(device, ptdtype, config, flash)
        self.text_projection = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        nn.init.normal_(self.text_projection, std=config.hidden_size ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def freeze_params(self):
        self.text_encoder.token_embedding.weight.requires_grad = False
        for i in range(self.freeze_layers):
            for param in self.text_encoder.layers[i].parameters():
                param.requires_grad = False

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        image_params = sum(p.numel() for p in self.image_encoder.parameters())
        other_params = total_params - text_params - image_params
        return total_params, text_params, image_params, other_params

    def forward(self, input_ids, images, infer=False):
        image = self.image_encoder(images)
        image = image.flatten(start_dim=2).permute(0, 2, 1)  # NCHW -> N(HW)C
        image = torch.cat([image.mean(dim=1, keepdim=True), image], dim=1)  # N(HW+1)C
        image = image + self.positional_embedding[None, :, :].to(image.dtype)  # N(HW+1)C
        image = self.img_attn(query=image[:, :1], key=image, value=image)
        image = image.reshape(-1, self.hidden_size)

        text = self.text_encoder(input_ids)
        text = text[:, -1, :] @ self.text_projection

        # normalized features
        image = image / image.norm(dim=1, keepdim=True)
        text = text / text.norm(dim=1, keepdim=True)

        if infer:
            return image @ text.t()

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image @ text.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(logits_per_image.shape[0]).to(logits_per_image.device)
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2 / self.norm_scale # normalization
        return loss

def retriever_clip(device, ptdtype, config, pretrained, model_path=None, retriever_path=None, vision_path=None, flash=True):
    model = CLIP(device, ptdtype, config, flash)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        replacer = 'module.'
        # replacer = '_orig_mod.'
        # if "module" == list(state_dict.keys())[0][:6]:
        #     replacer = 'module._orig_mod.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()})
    else:
        if config.vision == "resnet50_v15":
            state_dict = copy.deepcopy(model.image_encoder.state_dict())
            weights = torch.load(vision_path)
            for (k1, v1), (k2, v2) in zip(state_dict.items(), weights.items()):
                if v1.shape != v2.shape: break
                state_dict[k1] = weights[k2]
            model.image_encoder.load_state_dict(state_dict)
        else:
            vision_dict = torch.load(vision_path)['state_dict']
            model.image_encoder.load_state_dict({k.replace('module.', ''): v for k, v in vision_dict.items()}, strict=False)

        retriever_dict = torch.load(retriever_path)['state_dict']
        model.text_encoder.load_state_dict({k: v for k, v in retriever_dict.items()}, strict=False)
    return model
