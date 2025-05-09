import torch
import torch.nn as nn

class LightDehazeNetLite(nn.Module):
    def __init__(self):
        super(LightDehazeNetLite, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 4, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(4, 4, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(4, 4, 5, 1, 2, bias=True)

        self.e_conv4 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)  # reduced kernel size
        self.e_conv5 = nn.Conv2d(8, 4, 3, 1, 1, bias=True)

        self.e_conv6 = nn.Conv2d(8, 3, 3, 1, 1, bias=True)  # final layer to 3 channels

    def forward(self, x):
        conv1 = self.relu(self.e_conv1(x))
        conv2 = self.relu(self.e_conv2(conv1))
        conv3 = self.relu(self.e_conv3(conv2))

        concat1 = torch.cat((conv1, conv3), dim=1)  # [B, 8, H, W]

        conv4 = self.relu(self.e_conv4(concat1))
        conv5 = self.relu(self.e_conv5(conv4))

        concat2 = torch.cat((conv2, conv5), dim=1)  # [B, 8, H, W]
        conv6 = self.relu(self.e_conv6(concat2))

        # Final reconstruction formula
        out = self.relu((conv6 * x) - conv6 + 1)
        return out
