import timm
from timm.data import create_transform
from timm import create_model, list_models
from torch import nn
import torch
import os
 
    

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.flatten = flatten
    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
'''
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")
'''
    
class CFG:
    img_size = (1024, 1024)
    fc_dropout = 0.5
   
    model_name = "tf_efficientnetv2_s"#efficientnet_b2 resnetv2_50d_gn resnet200d dm_nfnet_f3 tf_efficientnetv2_s tf_efficientnetv2_b3 dm_nfnet_f0
    pretrained_weights = True
    num_classes = 1
    n_channels = 1


'''    
def BreastCancerModel(model_name,num_classes,n_channels):
    model = create_model(
        model_name, pretrained=CFG.pretrained_weights, in_chans=n_channels, num_classes=num_classes)
    return model
'''
class BreastCancerModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=1,
        n_channels=1,
    ):
        """
        Constructor.

        Args:
            model_name (timm model): Encoders' Name.
            num_classes (int, optional): Number of classes. Defaults to 1.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.model = create_model(model_name, pretrained=CFG.pretrained_weights,in_chans=CFG.n_channels, num_classes=CFG.num_classes,drop_rate=0.0, drop_path_rate = 0.0)
        
        self.model.global_pool = GeM(flatten=True)
#         self.global_pool = GeM(p_trainable=True)
        if(CFG.grad_ckpt):
            self.model.set_grad_checkpointing()
            
#         self.backbone_dim = self.model(torch.randn(1, 3, CFG.img_size[0], CFG.img_size[1])).shape[-1]
#         self.backbone_dim = self.model.feature_info[-1]['num_chs']
        
#         self.num_classes = num_classes
#         self.n_channels = n_channels
#         self.logits = nn.Linear(self.backbone_dim, self.num_classes)

    
        
#         self.fc_dropout = nn.ModuleList([
#                 nn.Dropout(CFG.fc_dropout) for _ in range(5)
#             ])

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_channels x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
        """
        
        x = self.model(x)
       
#         x = self.global_pool(x)
       
#         x = x[:,0,0,0]
      

#         for i, dropout in enumerate(self.fc_dropout):
#             if i == 0:
#                 logits = self.logits(dropout(x)).squeeze()
#             else:
#                 logits += self.logits(dropout(x)).squeeze()
        
#         logits /= len(self.fc_dropout)
     
        logits = x.squeeze()#x.squeeze()#self.logits(x).squeeze()
    
        return logits
        
def save_model(name, model, model_type=CFG.model_name):
    torch.save({'model': model.state_dict(), 'model_type': model_type}, f'{name}')
    
    

def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        model.load_state_dict(state_dict["model"], strict=strict)
    except BaseException:
        try:
            del state_dict['logits.weight'], state_dict['logits.bias']
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            del state_dict['encoder.conv_stem.weight']
            model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n")

    return model, state_dict["model_type"]
