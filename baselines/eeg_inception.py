import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        # 修改为适应空间维度的卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=(1, kernel_size), 
                                  padding=(0, padding), 
                                  groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EEGInception(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=3, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='relu', n_classes=2):
        super(EEGInception, self).__init__()
        
        # ============================= CALCULATIONS ============================= #
        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        
        # 调整尺度以适应1000个时间点
        scales_samples = [min(s, input_samples) for s in scales_samples]

        # ================================ INPUT ================================= #
        # 输入通道数设为1，因为输入形状是 (batch, 1, 3, 1000)
        self.input_layer = nn.Conv2d(1, ncha, kernel_size=(1, 1))

        # ========================== BLOCK 1: INCEPTION ========================== #
        b1_units = []
        for i in range(len(scales_samples)):
            unit = nn.Sequential(
                # 修改为适应通道数
                nn.Conv2d(ncha, ncha, kernel_size=(1, scales_samples[i]), padding="same"),
                nn.BatchNorm2d(ncha),
                nn.ELU(inplace=True),
                # 深度可分离卷积使用空间维度
                DepthwiseSeparableConv2d(ncha, filters_per_branch, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            b1_units.append(unit)

        self.b1_units = nn.ModuleList(b1_units)

        # ========================== BLOCK 2: INCEPTION ========================== #
        b2_units = []
        for i in range(len(scales_samples)):
            # 调整卷积核大小以适应空间维度
            kernel_height = min(3, int(scales_samples[i]/4))  # 最大不超过3
            unit = nn.Sequential(
                nn.Conv2d(filters_per_branch * len(scales_samples), filters_per_branch, 
                         kernel_size=(kernel_height, 1), 
                         padding=(kernel_height//2, 0)),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            b2_units.append(unit)

        self.b2_units = nn.ModuleList(b2_units)

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.b3_u1 = nn.Sequential(
            nn.Conv2d(filters_per_branch * len(scales_samples), 
                     int(filters_per_branch*len(scales_samples)/2), 
                     kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(int(filters_per_branch*len(scales_samples)/2)),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.b3_u2 = nn.Sequential(
            nn.Conv2d(int(filters_per_branch*len(scales_samples)/2), 
                     int(filters_per_branch*len(scales_samples)/4), 
                     kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(int(filters_per_branch*len(scales_samples)/4)),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(filters_per_branch*len(scales_samples)/4), n_classes)

    def forward(self, x):
        # ================================ INPUT ================================= #
        # 输入形状: (batch, 1, 3, 1000)
        x = self.input_layer(x)  # 输出: (batch, ncha, 3, 1000)

        # ========================== BLOCK 1: INCEPTION ========================== #
        b1_outputs = [unit(x) for unit in self.b1_units]
        b1_out = torch.cat(b1_outputs, dim=1)  # 在通道维度拼接
        
        # 空间平均池化
        b1_out = F.avg_pool2d(b1_out, (1, 4))  # 时间维度降采样

        # ========================== BLOCK 2: INCEPTION ========================== #
        b2_outputs = [unit(b1_out) for unit in self.b2_units]
        b2_out = torch.cat(b2_outputs, dim=1)  # 在通道维度拼接
        
        # 时间平均池化
        b2_out = F.avg_pool2d(b2_out, (1, 2))  # 时间维度进一步降采样

        # ============================ BLOCK 3: OUTPUT =========================== #
        b3_u1_out = self.b3_u1(b2_out)
        b3_u2_out = self.b3_u2(b3_u1_out)
        
        b3_out = self.avgpool(b3_u2_out)
        b3_out = b3_out.view(b3_out.size(0), -1)
        output = self.fc(b3_out)
        
        # return output
        return F.log_softmax(output, dim=1)

if __name__ == '__main__':
    # 创建适配 (batchsize, 1, 3, 1000) 的模型
    model = EEGInception(input_time=1000, fs=250, ncha=3, n_classes=2)
    
    # 测试输入
    data = torch.randn(32, 1, 3, 1000)  # batch_size=32, 1, 3通道, 1000时间点
    output = model(data)
    print(f"输入形状: {data.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")
    
    # 可选：安装 torchsummary 后查看模型结构。
    try:
        from torchsummary import summary
    except ImportError:
        summary = None
    if summary is not None:
        summary(model, (1, 3, 1000), device='cpu', batch_size=32)
