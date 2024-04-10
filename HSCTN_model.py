import torch.nn as nn
import torch
from net_modules import *
import numpy as np

channel_seq = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,13,17,18,21,23,24,27,29,30,33,35,36,39,16,20,19,22,26,25,28,32,31,34,38,37,40])-1).tolist()

class HSCTN(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 8), stride=(1, 4)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2)),
            nn.ELU(inplace=True),
        )
        self.adpavg = nn.AdaptiveAvgPool2d((4, 4))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.transformer = TransformerVit(dim=64, depth=3, heads=8, dim_head=256, mlp_dim=128, dropout=0.)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 1), stride=(4, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1)),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(64+64, 256, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.elu = nn.ELU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(256+64, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        input = x.unsqueeze(1)

        feature0 = input
        feature1_ = self.block1(feature0)
        ## atten


        atten_x = torch.mean(feature1_, dim=-1).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(atten_x.shape[0], -1, -1)
        atten_x = torch.cat((cls_tokens, atten_x), dim=1)
        tr_output = self.transformer(atten_x)
        cls_token = tr_output[:, 0, :]
        cls_token = self.elu(cls_token)

        feature1 = self.adpavg(feature1_)
        feature2 = self.block2(feature0)

        feature_cat = torch.cat((feature1, feature2), dim=1)
        feature = self.cat_conv(feature_cat)
        feature = torch.cat((feature, cls_token), dim=1)
        logits = self.classifier(feature)
        return logits



if __name__ == '__main__':
    cuda0 = torch.device('cuda:0')
    x = torch.rand((16, 40, 160), device=cuda0)
    model = HSCTN(n_class=3, channels=40, samples=160)
    model.cuda()

    output = model(x)
    print('output:', output.shape)




