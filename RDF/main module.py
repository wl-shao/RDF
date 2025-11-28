import torch
import torch.nn as nn
import torch.nn.functional as F

class FastRPMTransform(nn.Module):
    def __init__(self, start_idx, end_idx, output_size):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.output_size = output_size

    def forward(self, x):
        batch_size = x.shape[0]
        spectrum_roi = x[:, self.start_idx:self.end_idx+1]
        rpm_matrix = spectrum_roi.unsqueeze(2) - spectrum_roi.unsqueeze(1)
        rpm_normalized = (rpm_matrix - rpm_matrix.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]) \
                        / (rpm_matrix.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0] + 1e-8)
        rpm_normalized = rpm_normalized.unsqueeze(1)
        return F.interpolate(rpm_normalized, size=(self.output_size, self.output_size), mode='bilinear')

class BBSBranch(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 2, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(channels[0], channels[1], 3, 1, 1, groups=channels[1]),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class ADABranch(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 2, 3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class FeatureSelectionGate(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.importance_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_in, c_in//4),
            nn.ReLU(),
            nn.Linear(c_in//4, c_in),
            nn.Sigmoid()
        )
        self.fusion_controller = nn.Sequential(
            nn.Conv2d(c_in*2, c_in//2, 1),
            nn.ReLU(),
            nn.Conv2d(c_in//2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, bbs_feat, ada_feat):
        bbs_imp = self.importance_evaluator(bbs_feat)
        ada_imp = self.importance_evaluator(ada_feat)
        concat_feat = torch.cat([bbs_feat, ada_feat], dim=1)
        fusion_weights = self.fusion_controller(concat_feat)
        return bbs_feat * fusion_weights[:,0:1] + ada_feat * fusion_weights[:,1:2]

class ShotAdapter(nn.Module):
    def __init__(self, in_features, num_classes, dropout):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.adapter(x)

class RPM_SFNet(nn.Module):
    def __init__(self, num_classes, shots):
        super().__init__()
        self.rpm_transform = FastRPMTransform(49, 1043, 18)
        self.bbs_branch = BBSBranch(1, [24, 48, 96])
        self.ada_branch = ADABranch(1, [24, 48, 96])
        self.fusion_gate = FeatureSelectionGate(96)
        self.shot_adapters = nn.ModuleDict({
            str(s): ShotAdapter(96, num_classes, max(0.1, 0.3 - s/100))
            for s in shots
        })

    def forward(self, x, shot_size):
        x = self.rpm_transform(x)
        bbs_feat = self.bbs_branch(x)
        ada_feat = self.ada_branch(x)
        fused_feat = self.fusion_gate(bbs_feat, ada_feat)
        return self.shot_adapters[str(shot_size)](fused_feat)