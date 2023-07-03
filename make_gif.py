from PIL import Image
from tqdm import tqdm

exp_name = 'points'
label = 3

print(f"Creating GIF for Exp: {exp_name}, Label: {label}")

all_batch = []
for i in tqdm(range(50)):
    big_img = Image.open(f"exps/{exp_name}/rev_diffusion/label_{label}/{i:04d}.png")
    all_batch.append(big_img)

first_batch = all_batch[0]
first_batch.save(
    "our_awesome2.gif", 
    format="GIF", 
    append_images=all_batch,
    save_all=True, 
    duration=150, 
    # loop=0,
    optimize=True
)
