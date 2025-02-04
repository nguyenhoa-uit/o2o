
import torch
from IPython.display import display, Image
# from trl import DefaultDDPOStableDiffusionPipeline
from codes.modeling_sd_base import DefaultO2OStableDiffusionPipeline



# hugging: hoan17
# stabilityai/stablelm-2-1_6b
# stabilityai/stable-diffusion-3-medium
# stabilityai/stable-diffusion-2-1
pipeline = DefaultO2OStableDiffusionPipeline(
   "stabilityai/stable-diffusion-2-1",
)

model='base_Stable'
weight=None
weight='test_VSC'
loop=11

if weight !=None:
    model=weight

    pipeline.sd_pipeline.load_lora_weights(f"./outputs/{weight}")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# memory optimization

pipeline.vae.to(device, torch.float16)
pipeline.text_encoder.to(device, torch.float16)
pipeline.unet.to(device, torch.float16)
seed=0
torch.manual_seed(seed)
p="An extremely beautiful Asian girl"
q="Some girls are talking on grass"

# b="A yellowish tiger"
prompts = [p,p,p,q] 
results = pipeline(prompt=prompts,height=512,width=512)
for prompt, image in zip(prompts,results.images):
    loop+=1
    file_name=f"./outputs/images/seed{seed}_{model}_{prompt}_{loop}.png"
    image.save(file_name)
