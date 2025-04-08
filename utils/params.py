# Params 
class params: 
    def __init__(self): 
        self.root_dir = 'ACDC' 
        self.exp = 'SDCL' 
        self.model = 'unet' 
        self.pretrain_iterations = 800
        
        self.selftrain_iterations = 10 
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

        # Caussl parameters 
        self.consistency_type = 'mse'   
        self.max_step = 60 
        self.min_step = 60 
        self.start_step1 = 50 
        self.start_step2 = 50 
        self.cofficient = 3.0 
        self.max_iteration = 5000 
        self.thres_iteration = 20 