import torch.nn as nn
import copy

class EMAHelper():
    def __init__(self, mu=0.999):
        self.mu = mu
    
    def update_mu(self, new_mu):
        self.mu = new_mu

    def register(self, module):
        self.shadow = copy.deepcopy(module)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

    def update(self, module):
        shadow_para = dict(self.shadow.named_parameters())
        for name, param in module.named_parameters():
            if param.requires_grad:
                shadow_para[name].data = (1. - self.mu) * param.data + self.mu * shadow_para[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        used when resume training
        """
        self.shadow = state_dict
