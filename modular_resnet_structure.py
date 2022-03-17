import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class ResidualBlock(nn.Module):
    def __init__(self, in_featues, out_features, kernel_size, padding, stride=None):
        super(ResidualBlock, self).__init__()
        ## input_dims (batch_size, in_features, height, width)
        self.residual_connection = nn.Conv2d(in_channels=in_featues, out_channels=out_features, kernel_size=1)
        if in_featues == out_features:
            self.residual_connection = lambda x : x
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_featues, out_channels=out_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_features)
        )

    def forward(self, input):
        output = self.residual_connection(input) + self.block(input)
        return F.relu(output)

    
class Resnet(nn.Module):
    def __init__(self, lr, input_dims, in_featues, n_residual_blocks, \
        output_features: list,kernel_sizes: list, paddings: list, \
        strides: list, is_classifier=False, n_classes=None):
        super(Resnet, self).__init__()
        output_features = deque(output_features).appendleft(in_featues)
        tower = [ResidualBlock(output_features[i], output_features[i+1], kernel_sizes[i], \
                paddings[i], strides[i]) for i in range(n_residual_blocks)]
        self.residual_tower_list = nn.ModuleList(tower)
        self.residual_tower = nn.Sequential(*self.residual_tower_list)
        self.is_classifier = is_classifier
        if is_classifier:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(output_features[-1]*input_dims[-2]*input_dims[-1], n_classes)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        residual_output = self.residual_tower(input)
        if self.is_classifier:
            output = self.fc(residual_output)
        return output

