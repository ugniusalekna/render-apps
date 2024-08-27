import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, channels_in, channels_out, activation=True, batch_norm=True, **kwargs):
        layers = [nn.Conv2d(channels_in, channels_out, **kwargs)]
        layers += [nn.BatchNorm2d(channels_out)] if batch_norm else []
        layers += [nn.GELU()] if activation else []
        super().__init__(*layers)


class LinearBlock(nn.Sequential):
    def __init__(self, channels_in, channels_out, activation=True, dropout=0.0, **kwargs):
        layers = [nn.Linear(channels_in, channels_out, **kwargs)]
        layers += [nn.GELU()] if activation else []
        layers += [nn.Dropout(dropout)] if dropout > 0.0 else []
        super().__init__(*layers)


class DoodlesClassifier(nn.Module):
    def __init__(self, image_size, channels_in, hidden_layers, num_classes):
        super().__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        
        conv_layers = []
        
        for i, channels_out in enumerate(hidden_layers):
            stride = 2 if i == 0 or hidden_layers[i] != hidden_layers[i-1] else 1
            conv_layers.append(
                ConvBlock(channels_in, channels_out, kernel_size=3, padding=1, stride=stride)
            )
            channels_in = channels_out
        
        self.conv_blocks = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
    
        conv_output_size = self._get_conv_output_size(*image_size)
        
        self.fc_blocks = nn.Sequential(
            LinearBlock(conv_output_size, 64, dropout=0.25),
            LinearBlock(64, num_classes, activation=False)
        )
        
    @torch.no_grad()
    def _get_conv_output_size(self, height, width):
        self.eval()
        dummy_input = torch.zeros(1, 1, height, width)
        output = self.conv_blocks(dummy_input)
        self.train()
        return output.numel()
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.fc_blocks(x)
        return x