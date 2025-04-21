import torch 

def mask_image(img, mask_ratio = 0.5, block_size= 16): 
    batch_size, channel, H, W = img.shape 
    # assert H % block_size == 0 and W % block_size == 0, 'Block size is not suitable'

    h_blocks = H // block_size 
    w_blocks = W // block_size 
    total_blockes = h_blocks * w_blocks
    num_mask = int(total_blockes * mask_ratio)

    mask = torch.ones((batch_size, 1, H, W), device= img.device)

    for i in range(batch_size): 
        patch_indices = [(h, w) for h in range(h_blocks) for w in range(w_blocks)]
        selected = torch.randperm(len(patch_indices))[: num_mask]

        for idx in selected: 
            h_idx, w_idx = patch_indices[idx]
            top = h_idx * block_size 
            left = w_idx * block_size
            mask[i, :, top : top + block_size, left : left + block_size] = 0 
    
    masked_img = img * mask 
    return masked_img, mask