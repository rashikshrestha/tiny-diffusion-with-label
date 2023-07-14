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


    #! Define Noise Scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    model.eval()
    timesteps = list(range(len(noise_scheduler)))[::-1]

    labels = [0,1,2,3]
    # labels = [0.5,1.5,2.5]


    for label in labels:
        #! Get a random data sample
        sample = torch.randn(config.eval_batch_size, 2)
        initial_sample = sample.numpy()
        frames = []
        correct_filter = []

        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
            l = torch.from_numpy(np.repeat(label, config.eval_batch_size)).long()

            # print(t)
            # print(t.shape)
            # input()

            with torch.no_grad():
                sample = sample.to(config.device)
                t = t.to(config.device)
                l = l.to(config.device)
                residual = model(sample, t, l)
            sample = noise_scheduler.step(residual.cpu(), t[0].cpu(), sample.cpu())

            result = sample.numpy()

            #! Get expected values from the label
            lab = l[0].item() 

            if lab==0:
                exp_val = [2,2]
            elif lab==1:
                exp_val = [2,-2]
            elif lab==2:
                exp_val = [-2,-2]
            elif lab==3:
                exp_val = [-2,2]
            exp_val = np.array(exp_val)

            #! Error between exp_val and result
            error = np.abs(result - exp_val)

            # print("Error print:")
            # print(lab)
            # print(error)
            error_thres = 0.01
            correct_result = error < error_thres

            # print(correct_result)
            # input()

            both_correct_res = np.logical_and(correct_result[:,0], correct_result[:,1])
            correct_filter.append(both_correct_res)

            # both_incorrect_res = np.logical_not(both_correct_res)
            # correct_points = initial_sample[both_correct_res]
            # incorrect_points = initial_sample[both_incorrect_res]

            # print("Result ko shape:")
            # print(correct_points.shape)
            # print(incorrect_points.shape)
            # input()

            # if t[0].item() % 50 == 0:
            # all_correct.append(correct_points)
            # all_incorrect.append(incorrect_points)
            frames.append(result)

        #! ---------------------------------------------------------------------

        print(f"Saving reverse diffusion images for label {label} ...")
        imgdir = f"exps/{config.experiment_name}/rev_diffusion/label_{label}"
        os.makedirs(imgdir, exist_ok=True)

        frames = np.stack(frames)
        correct_filter = np.stack(correct_filter)

        
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6
        for i, frame in enumerate(frames):
            #! Plot forward diffusion
            plt.figure(figsize=(10, 10))
            plt.scatter(frame[:, 0], frame[:, 1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.savefig(f"{imgdir}/{i:04}.png")
            plt.close()

            #! Plot Convergence Plot
            both_correct_res = correct_filter[i]
            both_incorrect_res = np.logical_not(both_correct_res)
            correct_points = initial_sample[both_correct_res]
            incorrect_points = initial_sample[both_incorrect_res]

            plt.figure(figsize=(10, 10))
            plt.scatter(correct_points[:, 0], correct_points[:, 1], color='green', alpha=0.5)
            plt.scatter(incorrect_points[:, 0], incorrect_points[:, 1], color='red', alpha=0.5)
            plt.scatter(exp_val[0], exp_val[1], color='blue', s=200)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.savefig(f"{imgdir}/{i:04}-convergence.png")
            plt.close()            
            
            #! Histogram plot
            fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

            n_bins = 30
            # We can set the number of bins with the *bins* keyword argument.
            axs[0].hist(frame[:,0], bins=n_bins)
            axs[1].hist(frame[:,1], bins=n_bins)
            plt.savefig(f"{imgdir}/{i:04}-hist.png")
            plt.close()
            # print("Yo Frame ko shape:")
            # print(frame.shape)
            # input()
        print("Done !")