import argparse
from PIL import Image
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
from torch import nn
import numpy as np
import torch
import matplotlib
from lama_inpaint import inpaint_img_with_lama
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser(description='demo')
parser.add_argument("--image", type=str,
                    default="examples/case1.png", help="path to the image file")
# parser.add_argument("--out", type=str,
#                     default="examples/case1_out_test.png", help="path for the output file")
parser.add_argument("--seg_checkpoint", type=str,
                    default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024", help="checkpoint file")
parser.add_argument("--lama_config", type=str,
                    default="./lama/configs/prediction/default.yaml", help="path to the checkpoint file")
parser.add_argument("--lama_checkpoint", type=str,
                    default="./pretrained_models/big-lama", help="path to the checkpoint file")

args = parser.parse_args()
lama_config = './lama/configs/prediction/default.yaml'
lama_ckpt = './pretrained_models/big-lama'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matplotlib.use('TkAgg')

image = Image.open(args.image).convert("RGB")
image_w, image_h = image.size
img_new =image.resize((2048,1024))
img_new_array = np.array(img_new)

feature_extractor = SegformerFeatureExtractor.from_pretrained(args.seg_checkpoint)
model = SegformerForSemanticSegmentation.from_pretrained(args.seg_checkpoint)
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=True,
)

max_logits = upsampled_logits.max(dim=1)[0].squeeze(0)
max_logits_array = max_logits.detach().numpy()

# set thresh hold for max_logits_array
mask = np.copy(max_logits_array)
mask[mask < 2.5] = 1
mask[mask != 1] = 0

inpaint_img = inpaint_img_with_lama(img_new_array, mask, config_p=lama_config, ckpt_p=lama_ckpt, device=device)
inpaint_img_temp = Image.fromarray(inpaint_img)

# calculate the ssim between inpaint_img and img_new
ssim_score, ssim_img = ssim(inpaint_img, img_new_array, full=True, multichannel=True)












