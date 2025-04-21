def reconstruction_loss(X_rec, X_orig, mask, lam=0.1):
    """
    This loss inspired from: https://10.1007/s13042-024-02410-1
    """
    loss_masked = ((1 - mask) * (X_rec - X_orig) ** 2).sum()
    loss_visible = (mask * (X_rec - X_orig) ** 2).sum()
    
    total_pixels = X_orig.numel()
    loss = (loss_masked + lam * loss_visible) / total_pixels
    return loss