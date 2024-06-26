from rembg import remove, new_session
import folder_paths
from PIL import Image
import torch
import numpy as np
from functools import reduce
from tqdm import tqdm

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (MODEL_LIST, {"default": "u2net"}),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": None}),
                "alpha_matting_background_threshold": ("INT", {"default": None}),
                "alpha_matting_erode_size": ("INT", {"default": None}),
                "only_mask": ("BOOLEAN", {"default": False}),
                "post_process_mask": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "rembg"

    def remove_background(self, image, model_name, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size, only_mask, post_process_mask):
        session = new_session(model_name)
        img_list = []
        for img in tqdm(image, "Rembg"):
            sep_img = pil2tensor(remove(tensor2pil(img), session = session, alpha_matting = alpha_matting, alpha_matting_foreground_threshold = alpha_matting_foreground_threshold, alpha_matting_background_threshold = alpha_matting_background_threshold, alpha_matting_erode_size = alpha_matting_erode_size, only_mask = only_mask, post_process_mask = post_process_mask))
            img_list.append(sep_img)

        stack = reduce(lambda img1,img2: torch.cat([img1,img2]), img_list)

        return (stack,)


NODE_CLASS_MAPPINGS = {
    "Image Remove Background (rembg)": ImageRemoveBackgroundRembg
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Remove Background (rembg)": 'Rembg(Batch)'
}
MODEL_LIST = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam']