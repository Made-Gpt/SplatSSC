from mmengine.registry import Registry
GPD_LOSS = Registry('gpd_loss')

from .multi_loss import MultiLoss
from .ce_loss import CELoss, BCELoss
from .lovasz_loss import LovaszLoss, GlobalLovaszLoss
from .sem_geo_loss import Sem_Scal_Loss, Geo_Scal_Loss, Prob_Scal_Loss
from .focal_loss import FocalLoss
from .depth_loss import Depth_Scale_Loss, Depth_Huber_Loss, PCD_Huber_Loss, Depth_Gradient_Loss
