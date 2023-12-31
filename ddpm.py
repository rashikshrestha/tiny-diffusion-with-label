import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
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

    # dataset = datasets.get_dataset(config.dataset)
    # dataset = datasets.all_point_dataset()
    dataset = datasets.fn_dataset(labels=np.linspace(0,360,100).astype(np.int))

    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding).to(config.device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    losses = []

    print("Training model...")

    #! For each Epoch:
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        #! For each element in data loader:
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            # print(batch)
            # input()
            labels = batch[:, 2]
            batch = batch[:, :2]
            # print("Batch:")
            # print(batch)
            # print(labels)
            # input()

            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            #! Add noise to the Batch, for given timestep
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            noisy = noisy.to(config.device)
            timesteps = timesteps.to(config.device)
            labels = labels.to(config.device)

            #! Predict the noise added in the Batch
            # print(noisy.shape, timesteps.shape, labels.shape)
            noise_pred = model(noisy, timesteps, labels)
            # print(noise_pred.shape)noise_scheduler
            # input()

            #! MSE between Actual noise and Predicted noise
            noise = noise.to(config.device)
            loss = F.mse_loss(noise_pred, noise)

            #! Backward pass the loss
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(config.eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                l = torch.from_numpy(np.repeat(0, config.eval_batch_size)).long()

                # print(t)
                # print(t.shape)
                # input()

                with torch.no_grad():
                    sample = sample.to(config.device)
                    t = t.to(config.device)
                    l = l.to(config.device)
                    residual = model(sample, t, l)
                sample = noise_scheduler.step(residual.cpu(), t[0].cpu(), sample.cpu())

            #! Accumulate the Frame for the timestep 0
            frames.append(sample.numpy())

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
