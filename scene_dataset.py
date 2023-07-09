import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from read_write_model import read_images_text, qvec2rotmat, rotmat2qvec
import numpy as np
import torch


class SceneDataset(Dataset):
    def __init__(self, path='/home/menelaos/rashik/others/dtu_reconstruct'):
        self.path = path
        self.data = read_images_text(f"{path}/images.txt")
        # print(self.data)
        self.data_keys = list(self.data.keys())
        self.qctc, self.image_names = self.get_qc_tc_names()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # print(self.image_names)


    def get_qc_tc_names(self):
        qctc = []
        image_names = []
        for k in self.data_keys:
            data_point = self.data[k]

            image_names.append(data_point.name)

            q = data_point.qvec
            t = data_point.tvec.reshape(-1,1)
            R = qvec2rotmat(q)

            Rc = R.T
            tc = -R.T@t

            qc = rotmat2qvec(Rc) #TODO Check if these rot2mat and mat2rot works fine !!
            qc = qc / np.linalg.norm(qc)

            out_data = np.concatenate((qc, tc.flatten()))

            qctc.append(out_data)

        qctc = np.array(qctc)

        #! Normalize t
        x_min, x_max = np.min(qctc[:, 4]), np.max(qctc[:,4])
        y_min, y_max = np.min(qctc[:, 5]), np.max(qctc[:,5])
        z_min, z_max = np.min(qctc[:, 6]), np.max(qctc[:,6])

        qctc[:, 4] = (qctc[:, 4] - x_min) / (x_max - x_min)
        qctc[:, 5] = (qctc[:, 5] - y_min) / (y_max - y_min)
        qctc[:, 6] = (qctc[:, 6] - z_min) / (z_max - z_min)

        qctc[:, 4:] = (qctc[:, 4:] * 2) -1

        # print('min and max:')
        # print(x_min, y_min, z_min, x_max, y_max, z_max)
        # [-5.232566297776502, -3.9698998169604316, -1.132278253791447, 5.585892319151707, 3.1015755676630206, 2.2133326093127677]
        # print(np.max(qctc), np.min(qctc))

        return torch.as_tensor(qctc, dtype=torch.float32), image_names


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #! Get image
        img = Image.open(f"{self.path}/images/{self.image_names[idx]}")
        img_tensor = self.preprocess(img)

        #! Get qctc
        qctc = self.qctc[idx]

        # print(img_tensor.shape)
        # print(torch.min(img_tensor))
        # print(torch.max(img_tensor))

        #! Return
        return {
            'image': img_tensor,
            'qctc': qctc
        }


if __name__=='__main__':
    sd = SceneDataset()
    data = sd[2]

    print(data['image'].shape)
    print(data['qctc'])
