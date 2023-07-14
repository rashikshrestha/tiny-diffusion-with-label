import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from model import MLP
from noise_scheduler import NoiseScheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

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

    labels = [45, 135] #! Inputs
    expected_op = [[0.71, 0.71], [0.71, -0.71]] #! Expected Outputs

    ip_dim = 1
    op_dim = 2

    #! For each Label:
    for label, exp_op in zip(labels, expected_op):
        print("Labels and Expected Outputs are:", label, exp_op)
        #! Get a random data sample
        sample = torch.randn(config.eval_batch_size, op_dim)
        initial_sample = sample.numpy()

        fake_y = np.random.rand(*initial_sample.shape)

        frames = []
        correct_filter = []

        #! For each timestep:
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
            l = torch.from_numpy(np.repeat(label, config.eval_batch_size)).long()

            with torch.no_grad():
                sample = sample.to(config.device)
                t = t.to(config.device)
                l = l.to(config.device)
                residual = model(sample, t, l)

            sample = noise_scheduler.step(residual.cpu(), t[0].cpu(), sample.cpu())

            result = sample.numpy()

            #! Error between exp_val and result
            error = np.abs(result - exp_op)

            error_thres = 0.1
            correct_result = error < error_thres 

            correct_filter.append(correct_result)

            frames.append(result)

        #! ---------------------------------------------------------------------

        print(f"Saving reverse diffusion images for label {label} ...")
        imgdir = f"exps/{config.experiment_name}/rev_diffusion/label_{label}"
        os.makedirs(imgdir, exist_ok=True)

        # Convert to Numpy
        frames = np.stack(frames)
        correct_filter = np.stack(correct_filter)

        print("Correct Filter: ")
        print(correct_filter.shape)

        
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        for i, frame in enumerate(frames):
            n_bins = 30

            fig, (hist_ax, conv_ax) = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle(f"Input Label: {label}")


            #! Hist Plot 1
            hist_ax[0].set_xlim(-3,3)
            hist_ax[0].set_ylim(0, config.eval_batch_size)

            hist_ax[0].hist(frame[:,0], bins=np.linspace(-3,3,100))

            #! Hist Plot 2
            hist_ax[1].set_xlim(-3,3)
            hist_ax[1].set_ylim(0, config.eval_batch_size)

            hist_ax[1].hist(frame[:,1], bins=np.linspace(-3,3,100))



            #! For each output dim:
            for dim in range(op_dim):
                dim_data = initial_sample[:, dim]
                dim_fake_y = fake_y[:, dim]

                dim_fltr = correct_filter[i, :, dim]

                correct_points = dim_data[dim_fltr]
                incorrect_points = dim_data[np.logical_not(dim_fltr)]

                correct_rand_y = dim_fake_y[dim_fltr]
                incorrect_rand_y = dim_fake_y[np.logical_not(dim_fltr)]

                conv_ax[dim].set_xlim(-3,3)
                conv_ax[dim].set_ylim(0,1)


                conv_ax[dim].set_yticks([])

                conv_ax[dim].scatter(correct_points, correct_rand_y, color='green', alpha=0.5)
                conv_ax[dim].scatter(incorrect_points, incorrect_rand_y, color='red', alpha=0.5)

                conv_ax[dim].axvline(x = exp_op[dim], color = 'b', linewidth=4 , label = 'axvline - full height')

            # plt.ylim(ymin, ymax)
            plt.savefig(f"{imgdir}/{i:04}.png")
            plt.close()            
            
        print("Done !")