from typing import NamedTuple
from torch import Tensor
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt

to_tensor = T.ToTensor()
to_img = T.ToPILImage()

BACKBONE_ENUM = {
    'text': 'context_block',
    'latent': 'x_block'
}

def colormap_tensor(colormap, tensor: Tensor):
    if (colormap == 'none'):
        return tensor.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)

    # Normalize the tensor to the range [0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    tensor_np = tensor_normalized.numpy()

    colormap = plt.get_cmap(colormap)
    tensor_colored_np = colormap(tensor_np)
    tensor_colored = torch.from_numpy(tensor_colored_np)#tensor_colored_np[:, :, :3])
    return tensor_colored

matplotlib_colormaps = plt.colormaps()

class RenderAttentionSpot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd3_model": ("MODEL",),
                "joint_block": ("INT", {"default": 0, "max": 23}),
                "backbone": (["text", "latent"],),
                "view": (["query", "key", "value", "all-stacked", "all-interposed"],),
                "colormap": (["none", *matplotlib_colormaps],)
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"

    CATEGORY = "SD3 Power Lab/Visualize"

    def render(self, sd3_model, joint_block, backbone, view, colormap):
        km = sd3_model.model_state_dict()

        tensor_location = f'joint_blocks.{joint_block}.{BACKBONE_ENUM[backbone]}.attn.qkv.weight'
        attention_tensor: Tensor = None

        for k in km:
            if tensor_location in k:
                attention_tensor = km[k]
        
        if (attention_tensor is None):
            raise f"Could not locate attention tensor {tensor_location}"
        
        pre_image_tensor: Tensor = None
        q,k,v = (None,None,None)

        if (view == 'all-stacked'):
            pre_image_tensor: Tensor = attention_tensor
        elif (view == 'all-interposed'):
            pre_image_tensor=  attention_tensor.view(1536, 1536, 3)
        else:
            q,k,v = torch.split(attention_tensor, 1536)

        if (view == 'query'):
            pre_image_tensor = q
        elif (view == 'key'):
            pre_image_tensor = k
        elif (view == 'value'):
            pre_image_tensor = v

        if len(pre_image_tensor.shape) == 3:
            return (pre_image_tensor.unsqueeze(0), )
        else:
            return colormap_tensor(colormap, pre_image_tensor)

def calculate_modified_tensor(tensor_area: Tensor, tensor_mask: Tensor, operation, operation_value):
    modified_area = operation(tensor_area.clone()[tensor_mask], operation_value)


class AttentionToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd3_model": ("MODEL",),
                "joint_block": ("INT", {"default": 0, "max": 23}),
                "backbone": (["text", "latent"],),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"

    CATEGORY = "SD3 Power Lab/Hack"

    def render(self, sd3_model, joint_block, backbone):
        km = sd3_model.model_state_dict()

        tensor_location = f'joint_blocks.{joint_block}.{BACKBONE_ENUM[backbone]}.attn.qkv.weight'
        attention_tensor: Tensor = None

        for k in km:
            if tensor_location in k:
                attention_tensor = km[k]
        
        if (attention_tensor is None):
            raise f"Could not locate attention tensor {tensor_location}"

        return (attention_tensor.clone().view(1536, 1536, 3).unsqueeze(0),)

class ImageToAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd3_model": ("MODEL",),
                "joint_block": ("INT", {"default": 0, "max": 23}),
                "backbone": (["text", "latent"],),
                "attention_image": ("IMAGE",),
                "patch_strength": ("FLOAT", {"default": 1.0, "max": 1.0, "min": 0.0}),
                "model_strength": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0})
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "SD3 Power Lab/Hack"

    def patch(self, sd3_model, joint_block, backbone, attention_image, patch_strength, model_strength):
        m = sd3_model.clone()
        km = sd3_model.model_state_dict()

        tensor_location = f'joint_blocks.{joint_block}.{BACKBONE_ENUM[backbone]}.attn.qkv.weight'
        attention_tensor: Tensor = None

        key_to_patch = None

        for k in km:
            if tensor_location in k:
                attention_tensor = km[k]
                key_to_patch = k
        
        if (attention_tensor is None):
            raise f"Could not locate attention tensor {tensor_location}"
        
        modified_attention = attention_image.clone().squeeze(0).view(4608,1536)

        m.add_patches({key_to_patch: (modified_attention,)}, patch_strength, model_strength)

        return (m,)
        

NODE_CLASS_MAPPINGS = {
    "G370SD3PowerLab_RenderAttention": RenderAttentionSpot,
    "G370SD3PowerLab_AttentionToImage": AttentionToImage,
    "G370SD3PowerLab_ImageIntoAttention": ImageToAttention
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "G370SD3PowerLab_RenderAttention": "Render SD3 Attention",
    "G370SD3PowerLab_AttentionToImage": "SD3 Attention To Image",
    "G370SD3PowerLab_ImageIntoAttention": "SD3 Image Into Attention"
}
