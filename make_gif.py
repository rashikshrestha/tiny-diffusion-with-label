from PIL import Image
from tqdm import tqdm

exp_name = 'sin_cos_4'
label = 135

print(f"Creating GIF for Exp: {exp_name}, Label: {label}")

all_batch = []
for i in tqdm(range(50)):
    big_img = Image.open(f"exps/{exp_name}/rev_diffusion/label_{label}/{i:04d}.png")
    all_batch.append(big_img)

first_batch = all_batch[0]
first_batch.save(
    f"{exp_name}_{label}.gif", 
    format="GIF", 
    append_images=all_batch,
    save_all=True, 
    duration=200, 
    # loop=0,
    optimize=True
)
