import torch

input_path = '/home/menelaos/rashik/others/dtu_reconstruct/poses.txt'

names = []
qctcs = []
with open(input_path, "r") as file:
    data = file.readlines()

    for line in data:
        line_split = line[:-1].split(' ')
        name = line_split[0]

        qctc = []
        for i in range(1,8):
            qctc.append(float(line_split[i]))


        names.append(name)
        qctcs.append(qctc)

qctcs = torch.as_tensor(qctcs, dtype=torch.float32) 

print(names)
print(qctcs.shape)

        