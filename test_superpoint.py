from SuperGluePretrainedNetwork.models.superpointtrim import SuperPoint
from scene_dataset import SceneDataset
import torch

model = SuperPoint(config={})
dataset = SceneDataset(path='/home/menelaos/rashik/others/dtu_reconstruct', filename='poses_train_5.txt')

for d in dataset:
    img = d['image'][None]
    print(img.shape, torch.min(img), torch.max(img))

    desc = model(img)

    print(desc.shape)
    input()

