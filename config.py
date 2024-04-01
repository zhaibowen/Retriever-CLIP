from dataclasses import dataclass

@dataclass
class CLIPConfig_medium:
    gpu_num = 3
    batch_size = 64
    gradient_accumulation_steps = 1
    num_epoch = 16
    sequence_length = 32
    learning_rate = 3e-4
    min_lr = 6e-6
    vocab_size = 32000
    num_layers = 12
    hidden_size = 768
    num_heads = 12
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 800
    max_iters = 16000
    lr_decay_iters = 15000
    grad_clip = 1.0

    # vision = "resnet18"
    # res_layers = [2, 2, 2, 2]
    # res_channels = [64, 64, 128, 256, 512]
    # img_hidden_size = 512
    # img_num_head = 8

    vision = "resnet50_v15"
    res_layers = [3, 4, 6, 3]
    res_channels = [64, 256, 512, 1024, 2048]
    img_hidden_size = 2048
    img_num_head = 32

    img_size = 336
    spacial_dim = 11
