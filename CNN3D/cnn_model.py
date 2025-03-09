import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, in_channelss):
        super(Simple3DCNN, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU(),

                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU(),
            )

        # Define the convolutional layers
        self.encoder1 = conv_block(in_channels=1, out_channels=32)  # 16x16x440 -> 8x8x220
        self.encoder2 = conv_block(32, 64)  # 8x8x220 -> 4x4x110
        self.encoder3 = conv_block(64, 128)  # 4x4x110 -> 2x2x55
        # self.encoder4 = conv_block(128, 256)  # 2x2x55 -> 1x1x27

        self.fc1 = nn.Linear(128 * 2 * 2 * in_channelss, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Final output layer for a single value
        self.dropout = nn.Dropout(0.4)  # 50% dropout rate
        self.output_activation = nn.Sigmoid()  # or use nn.ReLU() depending on your range

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        x = self.encoder3(skip2)
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = nn.GELU()(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        x = self.output_activation(x)  # Adjusted based on the ground truth normalization
        return x  # Return the single predicted value
