import datasets
import torch

dnames = ['moons', 'dino', 'line', 'circle']

dataset_acc = []
for idx, name in enumerate(dnames):
    dataset = datasets.get_dataset(name)

    for d in dataset:
        data_point = [d[0][0].item(), d[0][1].item(), idx]
        dataset_acc.append(data_point)

all_dataset = torch.tensor(dataset_acc)
print(all_dataset.shape)

print(torch.min(all_dataset))
print(torch.max(all_dataset))