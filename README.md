# DeepDream

> :warning: **If you don't have a gpu with cuda, the style transfer execution time will be much longer**

# Prerequisites
Python >=3.8.10
# How to Install
```
sudo pip3 install -r requirements.txt 
```
# How to Use
if you are going to test with the example images, just run:
```
python3 main.py
```
# Results
The following images are results depending on the layer that is chosen as the final layer, the closer to the end of the network the more complex the shapes that appear, the first image is the result for the initial layers looking just simple shapes, the second deeper on the net with shapes that look like eyes and finally one of the last layers with several complex shapes that look like animals.

![download (1)](https://user-images.githubusercontent.com/32752004/216457096-d91ae7cb-5d79-43d4-8e3e-ab805ec0a84b.png)

![download (2)](https://user-images.githubusercontent.com/32752004/216457152-1d5790ab-a121-4374-94e6-1e0c943a8730.png)

![download (3)](https://user-images.githubusercontent.com/32752004/216457166-ff0a63fd-2103-4a38-88f7-18821f9893c9.png)


### Based on 
* [Pytorch monkeys a deep dream](https://www.kaggle.com/paultimothymooney/pre-trained-pytorch-monkeys-a-deep-dream)
