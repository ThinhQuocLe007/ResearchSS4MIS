import torch 
import torch.nn.functional as F 
import torch.nn as nn 

class DiceLoss(nn.Module): 
    def __init__(self, n_classes): 
        super(DiceLoss, self).__init__() 
        self.n_classes = n_classes
    
    def _one_hot_encoder(self, input_tensor): # torch.nn.functional.one_hot()
        """
        Apply one-hot encoder for input_tensor 
        Parameters: 
            - input_tensor.shape = (batchsize,1, H, W), the target image
        """
        tensor_list = [] 
        for i in range(self.n_classes): 
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim= 1)
        return output_tensor.float() 
    
    def _dice_loss(self, score, target): 
        target = target.float() 
        smooth = 1e-10 
        
        intersection = torch.sum(score * target)
        union = torch.sum(score* score) + torch.sum(target*target)
        dice = ( 2*intersection + smooth) / (union + smooth)
        loss = 1 - dice 
        return loss 
    
    def _dice_mask_loss(self, score, target, mask): 
        target = target.float() 
        mask = mask.float() 
        smooth = 1e-10 

        intersection = torch.sum(score * target * mask)
        union = torch.sum(score * score * mask ) + torch.sum(target * target * mask)
        dice = (2*intersection + smooth) / (union + smooth)
        loss = 1 - dice 
        return loss 

    def forward(self, inputs, target, mask= None, weight= None, softmax= False): 
        if softmax: 
            inputs = torch.softmax(inputs, dim= 1) 
        
        target = self._one_hot_encoder(target)

        # weight 
        if weight is  None: 
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), f'Predict.shape = {inputs.shape}, Target.shape= {target.shape}'
        class_wise_dice = [] 
        loss = 0.0 
        if mask is not None: 
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append( 1.0 - dice.item())
                loss += dice * weight[i]

        else: 
            for i in range(0, self.n_classes): 
                dice = self._dice_loss(inputs[:, i], target[:, i]) 
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i] 
        
        return loss / self.n_classes

def softmax_mse_loss(input_logits, target_logits): 
    """
    Take shoftmax on both sides and returns MSE loss
    
    """
    assert input_logits.size() == target_logits.size() 
    input_softmax = F.softmax(input_logits, dim= 1) 
    target_softmax = F.softmax(target_logits, dim= 1) 

    mse_loss = (input_softmax - target_softmax)**2 
    return mse_loss

def softmax_kl_loss(input_logits, target_logits): 
    """
    Take softmax on both sides and return KL divergence 
    """ 
    assert input_logits.size() == target_logits.size() 
    input_softmax = F.log_softmax(input_logits, dim= 1) 
    target_softmax = F.softmax(target_logits, dim= 1) 

    kl_div = F.kl_div(input_softmax, target_softmax, reduction='none')
    return kl_div

def l_correlation_cos_mean(model1, model2, linear_params1):
    """
    Compute CauSSL loss: https://doi.org/10.15223/policy-029
    Formula:
    loss_temp = (Va * qb) / ||Va|| * ||qB|| 
    L = 1/M (loss_temp * loss_temp)
    """ 
    total_loss = 0.0 
    count = 0 
    for name, parameter in model1.named_parameters(): 
        if 'conv' in name and 'weight' in name: 
            if len(parameter.shape) == 4: 
                w1 = parameter 
                w2 = model2.state_dict()[name] 
                w2 = w2.detach() 

                outdim = parameter.shape[0] # number of output channels 
                w1 = w1.view(outdim, -1) 
                w2 = w2.view(outdim, -1) 
                out = linear_params1[count](w2) # apply linear transform to w2 

                # normalization 
                out = nn.functional.normalize(out, dim=1) 
                w1_d = nn.functional.normalize(w1, dim=1) 

                loss_temp = torch.einsum('nc, nc->n', [out, w1_d]) # take cosine of norm variables so ||A|| = ||B|| = 1 
                total_loss += torch.mean(loss_temp * loss_temp) 

                count += 1 
    total_loss = total_loss / count
    return total_loss

def mix_loss(output, lab_a, lab_b, mask, l_weight=1.0, u_weight=0.5, unlab= False): 
    """
    Compute the L_BCP. Paper: 10.48550/arXiv.2305.00673
    Parameters: 
    - output: model output (shape: [B, C,H,w])
    - lab_a: ground truth for region A (shape: [B, H, W])
    - lab_b: ground truth for region B (shape: [B, H, W])
    - mask: binary mask to seperate region (shape: [B,H, W])
    - l_weight: weight for labeled part 
    - u_weight: weight for unlabeled parat 
    - unlab: Use pseudo label or not 
    """
    output_soft = F.softmax(output, dim= 1)
    lab_a, lab_b = lab_a.type(torch.int64), lab_b.type(torch.int64)

    weight_A, weight_B = l_weight, u_weight 
    if unlab: 
        weight_A, weight_B = u_weight, l_weight

    # Compute dice 
    mask_A = mask 
    mask_B = 1 - mask
    dice_loss_fn = DiceLoss(n_classes= 4)
    dice_A = dice_loss_fn(output_soft, lab_a.unsqueeze(1), mask_A.unsqueeze(1)) 
    dice_B = dice_loss_fn(output_soft, lab_b.unsqueeze(1), mask_B.unsqueeze(1)) 
    dice_loss = weight_A * dice_A + weight_B * dice_B

    # Compute CELoss 
    ce_loss_fn = nn.CrossEntropyLoss(reduction= 'none')
    ce_A = ce_loss_fn(output, lab_a) * mask_A 
    ce_B = ce_loss_fn(output, lab_b) * mask_B 

    # Norm 
    ce_A = ce_A.sum() / (mask_A.sum() + 1e-16)
    ce_B = ce_B.sum() / (mask_B.sum() + 1e-16)
    ce_loss = weight_A * ce_A + weight_B * ce_B
    return dice_loss, ce_loss
    

    



# def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
#     """
#     Compute BCP loss ( combine of CELoss and DiceLoss)
#     Paper: https://arxiv.org/abs/2305.00673
#     Parameters: 
#     - output (torch.Tensor): the output of model (logits)
#     - img_l (torch.Tensor): (multiply with mask)
#     - patch_l (): (multiply with 1 - mask ) 
#     - l_weight (float, default = 1.0): contribute factor of labeled data  
#     - u_weight (float, default = 0.5): the contribute factor of unlabeled data 
#     - unlab (boolean) 
#     """
#     CE = nn.CrossEntropyLoss(reduction='none')
#     dice_loss = DiceLoss(n_classes= 4)

#     img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
#     output_soft = F.softmax(output, dim=1)
#     image_weight, patch_weight = l_weight, u_weight
#     if unlab:
#         image_weight, patch_weight = u_weight, l_weight
#     patch_mask = 1 - mask
#     loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
#     loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
#     loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
#     loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
#     return loss_dice, loss_ce