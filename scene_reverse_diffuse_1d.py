import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from scene_dataset import SceneDataset
from scene_model import MLP
from noise_scheduler import NoiseScheduler
from read_write_model import read_images_text, qvec2rotmat, rotmat2qvec

def get_unit_cam():
    f = 10
    unit_cam = np.array([
        [0,0,0],
        [3,-2,f],
        [3,2,f],
        [-3,2,f],
        [-3,-2,f],
        [0,-4,f]
    ])

    seq = np.array([3,4,1,2,0,1,5,4,0,3,2])
    draw_cam = unit_cam[seq]

    return draw_cam

def get_cam_plot(qvec, tvec, unit_cam):
    R = qvec2rotmat(qvec)

    # t = -R.T@tvec.reshape(3,1)
    # R = R.T

    t = tvec
    cam = (R@unit_cam.T + t.reshape(3,1)).T
    return cam

def normalize_qt(qt):
        
    #! Normalize q
    q = qt[:, :4]
    q_norm = np.linalg.norm(q, axis=1)
    q = q/q_norm.reshape(-1,1)
    qt[:, :4] = q

    #! Normalize t
    #TODO: Should we do this as well ??
    # x_min, x_max = np.min(qt[:, 4]), np.max(qt[:,4])
    # y_min, y_max = np.min(qt[:, 5]), np.max(qt[:,5])
    # z_min, z_max = np.min(qt[:, 6]), np.max(qt[:,6])

    # qt[:, 4] = (qt[:, 4] - x_min) / (x_max - x_min)
    # qt[:, 5] = (qt[:, 5] - y_min) / (y_max - y_min)
    # qt[:, 6] = (qt[:, 6] - z_min) / (z_max - z_min)

    # qt[:, 4:] = (qt[:, 4:] * 2) -1

    return qt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    # parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    # parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    #! Define Dataset
    dataset = SceneDataset(test=True)


    #! Define Model
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding).to(config.device)

    model_path = f"exps/{config.experiment_name}/model.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model Loaded: {model_path}")
    model.eval()

    #! Define Noise Scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    #! Define Timesteps: 0 to 50
    timesteps = list(range(len(noise_scheduler)))[::-1]

    #! Define Inputs and Outputs
    # labels = [0,1,2,3] #! Inputs
    # expected_op = [[2,2],[2,-2],[-2,-2],[-2,2]] #! Expected Outputs

    # labels = [0, 90, 180, 270] #! Inputs
    # expected_op = [[0,1],[1,0],[0,-1],[-1,0]] #! Expected Outputs

    # labels = [45, 135] #! Inputs
    # expected_op = [[0.71, 0.71], [0.71, -0.71]] #! Expected Outputs
    # data = dataset[0]
    # labels = data['image']
    # expected_op = data['qctc']

    # print("got this from dataset:")
    # print(labels.shape, expected_op.shape)
    # input()



    ip_dim = 1
    op_dim = 7

    label_named = np.linspace(0,49, 49)

    unit_cam = get_unit_cam()*0.1

    #! For each Label:
    # for label, exp_op in zip(labels, expected_op):
    for count, dataa in enumerate(dataset):
        label = dataa['image']
        exp_op = dataa['qctc']
        print("Labels and Expected Outputs are:", label.shape, exp_op)
        #! Get a random data sample
        sample = torch.randn(config.eval_batch_size, op_dim)
        # sample = normalize_qt(sample)
        initial_sample = sample.numpy()

        fake_y = np.random.rand(*initial_sample.shape)

        frames = []
        correct_filter = []

        #! For each timestep:
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
            l = []
            for _ in range(config.eval_batch_size):
                l.append(label)
            l = torch.from_numpy(np.stack(l)).float()
            # print(l.shape)
            # l = torch.from_numpy(np.repeat(label, config.eval_batch_size)).long()

            with torch.no_grad():
                sample = sample.to(config.device)
                t = t.to(config.device)
                l = l.to(config.device)

                print('input shapes:')
                print(sample.shape, t.shape, l.shape)

                residual = model(sample, t, l)

            sample = noise_scheduler.step(residual.cpu(), t[0].cpu(), sample.cpu())
            # sample = normalize_qt(sample)

            result = sample.numpy()

            #! Error between exp_val and result
            error = np.abs(result - exp_op.numpy())

            error_thres = 0.1
            correct_result = error < error_thres 

            correct_filter.append(correct_result)

            frames.append(result)

        #! ---------------------------------------------------------------------

        # label 
        print(f"Saving reverse diffusion images for label {label_named[count]} ...")
        imgdir = f"exps/{config.experiment_name}/rev_diffusion/label_{label_named[count]}"
        os.makedirs(imgdir, exist_ok=True)

        # Convert to Numpy
        frames = np.stack(frames)
        correct_filter = np.stack(correct_filter)

        print("Correct Filter: ")
        print(correct_filter.shape)

        
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        print("Writing ...")
        for i, frame in tqdm(enumerate(frames)):

            # print("This is frame")
            # print(frame.shape)
            # input()

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=-70, azim=90, roll=0)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.set_zlim(-2,2)

            #! Normalize q
            frame = normalize_qt(frame)


            #! Plot all 100 camera poses in this time step
            for frm in frame:
                cam = get_cam_plot(frm[:4], frm[4:], unit_cam)
                ax.plot(cam[:,0], cam[:,1], cam[:,2])


            plt.savefig(f"{imgdir}/3d_{i:04}.jpg")
            plt.close()   
            #! -------------------------------
            # n_bins = 30

            # fig, (hist_ax, conv_ax) = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), gridspec_kw={'height_ratios': [2, 1]})
            # fig.suptitle(f"Input Label: {label_named[count]}")


            # #! Hist Plot 1
            # hist_ax[0].set_xlim(-3,3)
            # hist_ax[0].set_ylim(0, config.eval_batch_size)

            # hist_ax[0].hist(frame[:,4], bins=np.linspace(-3,3,100))

            # #! Hist Plot 2
            # hist_ax[1].set_xlim(-3,3)
            # hist_ax[1].set_ylim(0, config.eval_batch_size)

            # hist_ax[1].hist(frame[:,5], bins=np.linspace(-3,3,100))



            # #! For each output dim:
            # for dim in range(2):
            #     dim_data = initial_sample[:, dim+4]
            #     dim_fake_y = fake_y[:, dim+4]

            #     dim_fltr = correct_filter[i, :, dim+4]

            #     correct_points = dim_data[dim_fltr]
            #     incorrect_points = dim_data[np.logical_not(dim_fltr)]

            #     correct_rand_y = dim_fake_y[dim_fltr]
            #     incorrect_rand_y = dim_fake_y[np.logical_not(dim_fltr)]

            #     conv_ax[dim].set_xlim(-3,3)
            #     conv_ax[dim].set_ylim(0,1)


            #     conv_ax[dim].set_yticks([])

            #     conv_ax[dim].scatter(correct_points, correct_rand_y, color='green', alpha=0.5)
            #     conv_ax[dim].scatter(incorrect_points, incorrect_rand_y, color='red', alpha=0.5)

            #     conv_ax[dim].axvline(x = exp_op[dim+4], color = 'b', linewidth=4 , label = 'axvline - full height')

            # # plt.ylim(ymin, ymax)
            # plt.savefig(f"{imgdir}/{i:04}.png")
            # plt.close()            


        print("Done !")
        input()