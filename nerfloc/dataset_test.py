import datasets
import torch

torch.set_printoptions(sci_mode=False, precision=4)

# dnames = ['moons', 'dino', 'line', 'circle']

# dataset_acc = []
# for idx, name in enumerate(dnames):
#     dataset = datasets.get_dataset(name)

#     for d in dataset:
#         data_point = [d[0][0].item(), d[0][1].item(), idx]
#         dataset_acc.append(data_point)

# all_dataset = torch.tensor(dataset_acc)
# print(all_dataset.shape)

# print(torch.min(all_dataset))
# print(torch.max(all_dataset))

data = datasets.fn_dataset()

print(data.__len__())

print(data[0])
print(data[2000])
print(data[4000])
print(data[6000])


# for d in data:
    # print(d)
    # input()