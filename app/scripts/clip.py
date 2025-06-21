import sys
sys.path.append('src/blip')
sys.path.append('clip-interrogator')

import gradio as gr
from clip_interrogator import Config, Interrogator

config = Config()
config.blip_offload = True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64

ci = Interrogator(config)

async def inference(image, mode, best_max_flavors):
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)