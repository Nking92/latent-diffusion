from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
import torch
import numpy as np
import os
from base64 import b64encode
from uuid import uuid4
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()

class TextToImageRequest(BaseModel):
  caption: str
  ddim_steps: int = 100
  ddim_eta: float = 0.0
  # plms_sampling: bool = False
  height: int = 256
  width: int = 256
  batch_size: int = 1
  image_type: str = "" # Unused
  seed: str = "" # Unused


class TextToImageResponse(BaseModel):
  b64_images: List[str]


def load_model(verbose=False):
  config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
  ckpt = "models/ldm/text2img-large/model.ckpt"
  print(f"Loading model from {ckpt}")
  pl_sd = torch.load(ckpt, map_location="cpu")
  sd = pl_sd["state_dict"]
  model = instantiate_from_config(config.model)
  m, u = model.load_state_dict(sd, strict=False)
  if len(m) > 0 and verbose:
      print("missing keys:")
      print(m)
  if len(u) > 0 and verbose:
      print("unexpected keys:")
      print(u)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = model.to(device)
  # model.cuda()
  model.eval()
  return model

model = load_model()
sampler = DDIMSampler(model)
image_dir = "outputs/txt2img-samples"
os.makedirs(image_dir, exist_ok=True)

# Returns list of b64 encoded images
def generate_samples(options: TextToImageRequest) -> List[str]:
  images=[]
  with torch.no_grad():
    with model.ema_scope():
      uc = model.get_learned_conditioning(options.batch_size * [""])
      c = model.get_learned_conditioning(options.batch_size * [options.caption])
      shape = [4, options.height//8, options.width//8]
      samples_ddim, _ = sampler.sample(S=options.ddim_steps,
                                        conditioning=c,
                                        batch_size=options.batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=5.0,
                                        unconditional_conditioning=uc,
                                        eta=options.ddim_eta)

      x_samples_ddim = model.decode_first_stage(samples_ddim)
      x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

      for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        image = Image.fromarray(x_sample.astype(np.uint8))
        img_name = os.path.join(image_dir, f"{uuid4()}.png")
        image.save(img_name)
        with open(img_name, "rb") as f:
          images.append(b64encode(f.read()).decode("utf-8"))
  return images


@app.get("/")
def read_root():
  return {"Hello": "See /docs for available endpoints"}


@app.post("/txt2img")
def txt2img(req: TextToImageRequest):
  b64_images = generate_samples(req)
  return {"b64_images": b64_images}

