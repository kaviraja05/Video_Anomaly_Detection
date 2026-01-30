"""
I3D (Inflated 3D ConvNet) Model Implementation
Based on the paper: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D Max Pooling with 'SAME' padding (TensorFlow-style)."""
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    """Basic 3D convolution unit with batch normalization and activation."""
    
    def __init__(self, in_channels, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        super().__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias
        )
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """Inception module for I3D."""
    
    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + '/Branch_0/Conv3d_0a_1x1'
        )
        
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + '/Branch_1/Conv3d_0a_1x1'
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + '/Branch_1/Conv3d_0b_3x3'
        )
        
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + '/Branch_2/Conv3d_0a_1x1'
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + '/Branch_2/Conv3d_0b_3x3'
        )
        
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + '/Branch_3/Conv3d_0b_1x1'
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D architecture.
    
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf
    """
    
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, dropout_keep_prob=0.5,
                 extract_features=True):
        """
        Initialize I3D model.
        
        Args:
            num_classes: Number of output classes (400 for Kinetics)
            spatial_squeeze: Whether to squeeze spatial dimensions
            final_endpoint: Last endpoint to compute
            in_channels: Number of input channels (3 for RGB, 2 for flow)
            dropout_keep_prob: Dropout keep probability
            extract_features: If True, return 2048-D features instead of logits
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown final endpoint {final_endpoint}')

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.extract_features = extract_features
        self.logits = None

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(2, 2, 2),
            padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            480, [192, 96, 208, 16, 48, 64],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            512, [160, 112, 224, 24, 64, 64],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            512, [128, 128, 256, 24, 64, 64],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            512, [112, 144, 288, 32, 64, 64],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            528, [256, 160, 320, 32, 128, 128],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2],
            stride=(2, 2, 2),
            padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            832, [256, 160, 320, 32, 128, 128],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            832, [384, 192, 384, 48, 128, 128],
            name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits'
        )

        self.build()

    def replace_logits(self, num_classes):
        """Replace the final logits layer for fine-tuning."""
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits'
        )

    def build(self):
        """Build the model by registering all modules."""
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B: batch size
               C: channels (3 for RGB)
               T: number of frames (typically 16)
               H, W: spatial dimensions (typically 224x224)
        
        Returns:
            If extract_features=True: 2048-D feature vector
            Else: class logits
        """
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
            
            if end_point == 'Mixed_5c' and self.extract_features:
                # Global average pooling to get 1024-D features
                # Shape: (B, 1024, T', H', W') -> (B, 1024)
                features = F.adaptive_avg_pool3d(x, (1, 1, 1))
                features = features.view(features.size(0), -1)
                return features

        # Regular forward pass for classification
        x = self.avg_pool(x)
        if self.logits is not None:
            x = self.dropout(x)
            x = self.logits(x)
            if self._spatial_squeeze:
                x = x.squeeze(3).squeeze(3)
        x = x.mean(2)
        return x

    def extract_features_from_video(self, x):
        """
        Extract 1024-D features from video clips.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            features: Tensor of shape (B, 1024)
        """
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
            
            if end_point == 'Mixed_5c':
                break
        
        # Global average pooling
        features = F.adaptive_avg_pool3d(x, (1, 1, 1))
        features = features.view(features.size(0), -1)
        return features


def load_i3d_model(weights_path=None, device='cpu'):
    """
    Load I3D model with optional pretrained weights.
    
    Args:
        weights_path: Path to pretrained weights (.pt or .pth file)
        device: Device to load model on
    
    Returns:
        model: I3D model in eval mode
    """
    model = InceptionI3d(
        num_classes=400,
        in_channels=3,
        extract_features=True
    )
    
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded I3D weights from {weights_path}")
    
    model = model.to(device)
    model.eval()
    return model
