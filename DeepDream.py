import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import transforms
import PIL
import scipy.ndimage as nd
import PIL.Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    
normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalise_resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def tensor_to_img(t):
    a = t.numpy()
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
    inp *= 255
    inp = np.uint8(np.clip(inp, 0, 255))
    return PIL.Image.fromarray(inp)

def load_image(path, size=[1024, 1024]):
    img = PIL.Image.open(path)
    img.thumbnail(size, PIL.Image.ANTIALIAS)        
    img_tensor = normalise(img).unsqueeze(0)
    return  img_tensor.numpy()

def deepdream(model, base_img,
              octave_n=6, octave_scale=1.4, iter_n=10,
              control=None, step_size=1.5, end=28, jitter=32):
    octaves = [base_img]
    
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)
        
        src = octave_base + detail
        
        for i in range(iter_n):
            src = step_fn(model, src, control, step_size, end, jitter)

        detail = src.numpy() - octave_base

    return src

def filter_step(model, img, layer_index, filter_index, step_size=5, display=True, use_L2=False):
    
    model.zero_grad()
    
    img_var = Variable(torch.Tensor(img).to(device), requires_grad=True)
    optimizer = SGD([img_var], lr=step_size, weight_decay=1e-4)
    
    
    for index, layer in enumerate(model.features):
        img_var = layer(img_var)
        if index == layer_index:
            break

    output = img_var[0, filter_index]
    loss = output.norm() #torch.mean(output)
    loss.backward()
    
    if use_L2:
        #L2 normalization on gradients
        mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2) + 1e-5])
        mean_square = mean_square.to(device)
        img_var.grad.data /= torch.sqrt(mean_square)
        img_var.data.add_(img_var.grad.data * step_size)
    else:
        optimizer.step()
    
    result = img_var.data.cpu().numpy()
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std)
    
    
    return torch.Tensor(result)

def objective(dst, guide_features):
    if guide_features is None:
        return dst.data
    else:
        x = dst.data[0].cpu().numpy()
        y = guide_features.data[0].cpu().numpy()
        ch, w, h = x.shape
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        A = x.T.dot(y)
        diff = y[:, A.argmax(1)]
        diff = torch.Tensor(np.array([diff.reshape(ch, w, h)])).to(device)
        return diff

def step_fn(model, img, control=None, step_size=1.5, end=28, jitter=32):

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    
    img = np.roll(np.roll(img, ox, -1), oy, -2)
    
    
    img_var = Variable(torch.Tensor(img).to(device), requires_grad=True)
    model.zero_grad()
      
    x = img_var
    for index, layer in enumerate(model.features.children()):
        x = layer(x)
        if index == end:
            break
    
    delta = objective(x, control)
    x.backward(delta)
    
    #L2 Regularization on gradients
    mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2)])
    mean_square = mean_square.to(device)
    img_var.grad.data /= torch.sqrt(mean_square)
    img_var.data.add_(img_var.grad.data * step_size)
    
    result = img_var.data.cpu().numpy()
    result = np.roll(np.roll(result, -ox, -1), -oy, -2)
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std)
    
    return torch.Tensor(result)
                                                             
