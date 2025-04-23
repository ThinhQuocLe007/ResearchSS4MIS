# Params 
class params: 
    def __init__(self): 
        self.root_dir = 'ACDC' 
        self.model = 'unet' 
        self.pretrain_iterations = 2000
        self.selftrain_iterations = 500
        self.batch_size = 24
        self.deterministic = 1 # What the fucck here
        self.base_lr = 1e-3
        self.patch_size = [256,256] 
        self.seed = 42 
        self.num_classes = 4 

        # label and unlabel 
        self.labeled_bs = 12
        self.label_num = 7 
        self.u_weight = 0.5 

        # Cost 
        self.gpu = '0' 
        self.consistency = 0.1
        self.consistency_rampup = 200.0 
        self.magnitude = '6.0' 
        self.s_param = 6

        # MBCP 
        self.bcp_weight = 1.0 # iherit: https://doi.org/10.1007/s13042-024-02410-1
        self.recon_weight = 1.0 