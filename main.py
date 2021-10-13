import torch
from torchvision import models
from DeepDream import *


inputImage = "image_sample/galaxias.jpg"
model = models.vgg19(pretrained=True) 
# model = models.vgg16(pretrained=True) 
# model = models.densenet121(pretrained=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_np = load_image(inputImage, size=[1024, 1024])

dream = deepdream(model, input_np, end=28, step_size=0.06, octave_n=6)
dream = tensor_to_img(dream)
dream.save('image_sample/vgg19_.jpg')