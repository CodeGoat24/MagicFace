
import cv2, os
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.io import read_image
from matplotlib import colormaps
from PIL import Image
from einops import rearrange
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def load_image(image_path, device):
    """
    Load an image, preprocess it, and prepare it for use in a deep learning model.

    Parameters:
        image_path (str): The file path to the image.
        device (torch.device): The PyTorch device (CPU or GPU) on which the image should be loaded.

    Returns:
        torch.Tensor: A PyTorch tensor representing the preprocessed image.

    Example:
        >>> image = load_image('example.jpg', device='cuda')
    """
    # Load the image
    image = read_image(image_path)  # You should have a read_image function to load the image

    # Preprocess the image
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # Normalize pixel values to the range [-1, 1]
    image = F.interpolate(image, (512, 512))  # Resize the image to a specified size

    # Move the preprocessed image to the specified PyTorch device
    image = image.to(device)

    return image

def load_mask(mask_path, device, size=(128, 128), mode='nearest'):
    """
    Load an image mask, resize it, and prepare it for use in PyTorch [0, 255] -> [0, 1], returned shape (1,1,H,W)
    Parameters:
        mask_path (str): The file path to the image mask.
        device (torch.device): The PyTorch device (CPU or GPU) on which the mask should be loaded.
        size (tuple, optional): The target size to which the mask should be resized. Default is (64, 64).
        mode (str, optional): The interpolation mode for resizing. Options include 'nearest', 'bilinear', 'bicubic', and more.
            Default is 'nearest'.

    Returns:
        torch.Tensor(1,1,H,W): A PyTorch tensor with shape (1, 1, H, W) representing the resized image mask.

    Example:
        >>> mask = load_mask('mask.png', device='cuda', size=(128, 128), mode='nearest')
    """
    mask = read_image(mask_path)
    mask = F.interpolate(mask.unsqueeze(0), size=size, mode=mode)
    mask = (mask / 255.).to(torch.uint8).to(device)
    return mask

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255*cam)


def visualize_attention_map(attn, viz_cfg, cur_step, cur_att_layer, index=None):
    """
    Visualize the attention map only in the multi-reference self-attention (MRSA) module.
    Input:
        viz_cfg: includes generated image and reference images info.
        cur_step: viz at current denoising step.
        cur_att_layer: viz at current attention layer.
    """

    image_paths = [viz_cfg.generated_image_path]
    # set the dir to save all attention map
    if index is None:
        attn_map_save_dir = os.path.join(viz_cfg.results_dir, 'attention_maps', f'step_{cur_step}, layer_{cur_att_layer}')
        image_paths = image_paths + [list(viz_cfg.ref_image_infos)]
    else:
        attn_map_save_dir = os.path.join(viz_cfg.results_dir, 'attention_maps', f'step_{cur_step}, layer_{cur_att_layer}, index_{index}')
        image_paths = image_paths + [list(viz_cfg.ref_image_infos)[index]]
    os.makedirs(attn_map_save_dir, exist_ok=True)
    
    save_res = 512 # save attention map to the specified image size
    H = W = cur_res = int(attn.shape[1] ** 0.5) # resolution of current feature(q,k,v)

    # set image paths includes generated image and input images
    

    
    
    generated_image = cv2.resize(cv2.imread(viz_cfg.generated_image_path), (save_res, save_res), interpolation=cv2.INTER_NEAREST)

    attn_mean = attn.mean(0) # Merge the attention of 8 heads, attn_mean shape: (H*W, H*W)
    # visualize the attention maps, for simplicity, we dont viz all of them
    for i in range(0, attn_mean.shape[0], 10): 
        # We have H*W query pixels and corresponding H*W attention map, the i-th is the index of the current query pixel.
        pixels = np.zeros(H * W , dtype=np.uint8)
        pixels[i] = 1
        pixels = pixels.reshape(H, W)
        pixels = cv2.resize(pixels, (save_res, save_res), interpolation=cv2.INTER_NEAREST)
        # set the query pixel within the generated image to white color
        query_pixel_on_generated_image = generated_image.copy()
        query_pixel_on_generated_image[pixels > 0] = [255, 255, 255] 
        
        # processing rgb images and attention maps
        images = []
        attention_maps = []
        for j, image_path in enumerate(image_paths):
            attn_part = attn_mean[i][H*W*j: H*W*(j+1)]
            attention_map = attn_part.reshape(cur_res, -1).clone().cpu().numpy()

            image = cv2.resize(cv2.imread(image_path), (save_res, save_res), interpolation=cv2.INTER_NEAREST)
            attention_map = cv2.resize(attention_map, (save_res, save_res), interpolation=cv2.INTER_LINEAR)
            
            images.append(image)
            attention_maps.append(attention_map)

        
        # show multi reference self attention map on the generated image and input images
        images = np.concatenate(images, axis=1)
        attention_maps = np.concatenate(attention_maps, axis=1)
        attn_maps_on_images = show_cam_on_image(images / 255., attention_maps / attention_maps.max(), use_rgb=False, colormap=cv2.COLORMAP_JET)
        
        # save the generated image and attention maps
        curr_attn_map = np.concatenate((query_pixel_on_generated_image, attn_maps_on_images), axis=1)
        cv2.imwrite(os.path.join(attn_map_save_dir, f'{i}.png'), curr_attn_map)
        
def visualize_correspondence(viz_cfg, attn, cur_step, cur_att_layer, index=None):
    """
    Visualize the feature correspondence in the multi-reference self-attention (MRSA) module.
    Input:
        viz_cfg: includes generated image and reference images info.
        cur_step: viz at current denoising step.
        cur_att_layer: viz at current attention layer.
    """
    cur_res  = int(attn.shape[1] ** 0.5) # current resolution of q,k,v feature
    
    attn_mean = attn.mean(0).cpu().numpy() 
    attn_refs = attn_mean / attn_mean.max()
    ref_num   = attn_refs.shape[-1] // (cur_res * cur_res)  # number of input reference images + 1
    
    # providing a mask for the generated combined concept is suggested
    if viz_cfg.generated_mask_path == '':
        mask_gen = np.ones((cur_res,cur_res), dtype=int)
    else:
        mask_gen = cv2.imread(viz_cfg.generated_mask_path, -1) // 255

    def polar_color_map(image_size, mask=None):
        """
        generate a square polar colormap.
        """
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        coords = np.stack((x, y), axis=-1)
        
        angle = (np.arctan2(coords[..., 1] - image_size // 2, coords[..., 0] - image_size // 2) + np.pi) % (2 * np.pi) # calculate angle
        normalized_angle = angle / (2 * np.pi) # normalize angle
        
        cmap = colormaps['hsv_r'] # get the polar colormap values
        bgr_colors = (cmap(normalized_angle)[:, :, :3] * 255).astype(np.uint8)
    
        hsv_colors = cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2HSV)

        saturation_factor = 0.5  # control the level of saturation
        hsv_colors[..., 1] = hsv_colors[..., 1] * saturation_factor

        bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)
        rgb_colors = cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2RGB)

        if mask is not None:
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            rgb_colors[mask == 0] = [255, 255, 255]

        return rgb_colors
    
    colormap = polar_color_map(cur_res, mask_gen) # a square polar color map

    max_attn_value_indices = np.argmax(attn_refs, axis=0)  # the index of the maximum attention value corresponding to each query feature
    corrspondence = colormap.reshape(-1, 3)[max_attn_value_indices] # core of this function, pick up pixel color from the colormap according to the the index of the maximum attention value.
    corrspondence = corrspondence.reshape(-1, cur_res, 3)
    corrspondence = np.concatenate(np.split(corrspondence, ref_num)[1:], axis=1) # [1:] means that we only visualize the correspondence betweent the generated image and the reference images.
    
    save_res = 512
    cmap_resize = cv2.resize(colormap, (save_res, save_res), interpolation=cv2.INTER_NEAREST)
    corr_resize = cv2.resize(corrspondence, (save_res*(ref_num-1), save_res), interpolation=cv2.INTER_NEAREST)
    cmap_corr_resize = np.concatenate((cmap_resize, corr_resize), axis=1)

    corr_save_dir = os.path.join(viz_cfg.results_dir, 'feature_correspondences')
    os.makedirs(corr_save_dir, exist_ok=True)
    if index is None:
        cv2.imwrite(os.path.join(corr_save_dir, f'step_{cur_step}, layer_{cur_att_layer}.png'), cmap_corr_resize)
    else:
        cv2.imwrite(os.path.join(corr_save_dir, f'step_{cur_step}, layer_{cur_att_layer}, index_{index}.png'), cmap_corr_resize)


def save_mask_data(ca_mask, fore_mask, prompt, output_dir, step, R):


    mask_t  =F.interpolate(ca_mask, scale_factor=R, mode = 'nearest')
    mask_fore  =F.interpolate(fore_mask, scale_factor=R, mode = 'nearest')


    # save foreground mask
    mask_fore = mask_fore[0] * 255  
    mask_fore = mask_fore.type(torch.uint8) 
    
    mask_fore = mask_fore.cpu().numpy()  
    
    mask_fore = mask_fore.squeeze(0)

    image = Image.fromarray(mask_fore)  
    
    output_dir_fore = os.path.join(output_dir, 'fore_masks')
    os.makedirs(output_dir_fore, exist_ok=True)
    image.save(os.path.join(output_dir_fore, f'fore_mask_{step}.jpg'))

    # save rendered mask

    mask_list = mask_t[0][1:len(prompt[0].split())+1]

    value = 0  # 0 for background

    mask_all = torch.ones(mask_list.shape[-2:])
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy() == 1] = value + idx + 1

        mask*= 255 
        mask = mask.type(torch.uint8) 
        
        mask = mask.cpu().numpy()  
        
        image = Image.fromarray(mask)  
         
        prompt_dir = os.path.join(output_dir, 'word_mask')
        prompt_dir = os.path.join(prompt_dir, f'step_{step}')
        os.makedirs(prompt_dir, exist_ok=True)
        image.save(os.path.join(prompt_dir, f'mask_{prompt[0].split()[idx]}.jpg'))
 
    output_dir = os.path.join(output_dir, 'latent_masks')
    os.makedirs(output_dir, exist_ok=True)


    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'mask_{step}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

 