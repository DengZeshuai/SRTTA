import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Classifier(nn.Module):
    def __init__(self, n_cls=3, feat_dim=128):
        super(Classifier,self).__init__()
        self.class_num = n_cls
        self.loss = nn.BCELoss(reduction="mean")
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.head = nn.Linear(1000, feat_dim)
        self.classifier = nn.Linear(feat_dim, self.class_num)

    def forward(self, x):
        x = F.normalize(self.head(self.backbone(x)), dim=1)
        res = self.classifier(x)
        return res
    
    def computeLoss(self, pred, gt):
        loss = self.loss(pred.sigmoid(), gt)
        return loss