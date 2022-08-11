from turtle import pos
import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2))

    def forward_once(self, x):
        output = x
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) (1/2) * torch.pow(euclidean_distance, 2) +
                                      (label) (1/2)* torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# class TripletLoss(torch.nn.Module):
#     def __init__(self, margin = 2.0):
#         super(TripletLoss,self).__init__()
#         self.margin = margin
    
    # def forward(self, anchor, positive, negative):

    #     pos_dist = F.pairwise_distance(anchor, positive)
    #     neg_dist = F.pairwise_distance(anchor, negative)
    #     triplet_loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
    #     return triplet_loss
    # def forward(self, anchor, positive, negative):
    #     distance_post = (anchor - positive).pow(2).sum(1)
    #     distance_neg = (anchor - negative).pow(2).sum(1)
    #     losses = F.relu