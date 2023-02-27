from transformers import DPTForSemanticSegmentation
from PIL import Image
import torch
import requests
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/raw/main/ADE_val_00000001.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image

net_h = net_w = 480

transform = Compose([
        Resize((net_h, net_w)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

pixel_values = transform(image).unsqueeze(0)

model = model.cuda()
pixel_values = pixel_values.cuda()

with torch.no_grad():
  outputs = model(pixel_values)
  logits = outputs.logits

logits.shape

prediction = torch.nn.functional.interpolate(
                logits, 
                size=image.size[::-1], 
                mode="bicubic", 
                align_corners=False
            )
prediction = torch.argmax(prediction, dim=1) + 1
prediction = prediction.squeeze().cpu().numpy()