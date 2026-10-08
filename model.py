import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet50
from gen2 import Generator
from SEM import GradientComputation4
from LSM import LSM
from Net import Net


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, 8)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x
        x = self.ca(x) * x
        rub1 = out - x
        x = self.sa(x) * x
        rub2 = out - x
        return x + out, rub1, rub2


class VehicleStructurePrototype(nn.Module):
    def __init__(self, dim=2048, proto_num=6):
        super(VehicleStructurePrototype, self).__init__()
        self.proto_num = proto_num
        self.dim = dim
        
        self.learnable_proto = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(proto_num, dim), mode='fan_out'))
        
        self.proto_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        
        self.scale = dim ** -0.5
        
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim))
        
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        proto = self.proto_norm(self.learnable_proto)  # [proto_num, C]
        feat = self.feat_norm(x_flat)  # [B, HW, C]
        
        sim = torch.matmul(proto, feat.permute(0, 2, 1))  # [B, proto_num, HW]
        sim = sim * self.scale
        sim = F.softmax(sim, dim=-1)
        
        part_feat = torch.matmul(sim, x_flat)  # [B, proto_num, C]
        part_feat = self.proj(part_feat)  # [B, proto_num, C]
        
        part_feat_avg = part_feat.mean(dim=1)  # [B, C]
        
        global_feat = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)  # [B, C]
        
        fused_feat = global_feat + 0.3 * part_feat_avg  # [B, C]
        fused_feat = self.dropout(fused_feat)
        
        return fused_feat, sim, part_feat


class FusionModel(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50', proto_num=6):
        super(FusionModel, self).__init__()
        self.thermal_module = ThermalModule(arch=arch)
        self.visible_module = VisibleModule(arch=arch)
        self.base_resnet = BaseResNet(arch=arch)
        self.non_local = no_local
        self.genA2B = Generator(input_nc=200)
        self.SEM4 = GradientComputation4()
        self.LSM = LSM()
        
        self.cbam = CBAM(2048)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(16, 16))
        
        from torchvision.models.resnet import Bottleneck
        self.p2_0 = Bottleneck(1024, 512, stride=1,
                              downsample=nn.Sequential(
                                  nn.Conv2d(1024, 2048, 1, stride=1, bias=False),
                                  nn.BatchNorm2d(2048)))
        self.p2_1 = Bottleneck(2048, 512)
        self.p2_2 = Bottleneck(2048, 512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(16, 16))
        
        self.layer3_adjust = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.layer3_adjust.weight, mode='fan_in')
        
        self.enhi_weight = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.idfr_residual_weight = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        
        # 车辆结构原型模块（VSP）
        self.VSP = VehicleStructurePrototype(dim=2048, proto_num=proto_num)
        self.vsp_weight = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        
        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1 = nn.Linear(256, pool_dim, bias=False)
        self.classifier2 = nn.Linear(512, pool_dim, bias=False)
        self.classifier3 = nn.Linear(1024, pool_dim, bias=False)
        self.classifier4 = nn.Linear(2048, pool_dim, bias=False)
        
        self._init_weights()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def _init_weights(self):
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)
        self.classifier3.apply(weights_init_classifier)
        self.classifier4.apply(weights_init_classifier)

    def forward(self, x1, x01, x2=None, x02=None, modal=0):
        x_fc = None
        if modal == 0:
            x1, x01 = self.visible_module(x1, x01)
            x2, x02 = self.thermal_module(x2, x02)
            x = torch.cat((x1, x2), 0)
            x0 = torch.cat((x01, x02), 0)
            x, x_fc = self.LSM(x, x0)
        elif modal == 1:
            x1, x01 = self.visible_module(x1, x01)
            x, _ = self.LSM(x1, x01)
        elif modal == 2:
            x1, x01 = self.thermal_module(x1, x01)
            x, _ = self.LSM(x1, x01)
        
        real_layer3 = None
        
        if self.non_local == 'on':
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
            fc_1 = self.classifier1(x.permute(0, 2, 3, 1))
            fc_1 = self.classifier(fc_1)
            x, cam_1 = self.genA2B(x, fc_1, self.training)
            x = self.SEM4(x, cam_1)
            
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
            fc_2 = self.classifier2(x.permute(0, 2, 3, 1))
            fc_2 = self.classifier(fc_2)
            x, cam_2 = self.genA2B(x, fc_2, self.training)
            x = self.SEM4(x, cam_2)
            
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
            real_layer3 = x
            fc_3 = self.classifier3(x.permute(0, 2, 3, 1))
            fc_3 = self.classifier(fc_3)
            x, cam_3 = self.genA2B(x, fc_3, self.training)
            x = self.SEM4(x, cam_3)
            
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
            fc_4 = self.classifier4(x.permute(0, 2, 3, 1))
            fc_4 = self.classifier(fc_4)
            x, cam_4 = self.genA2B(x, fc_4, self.training)
            x = self.SEM4(x, cam_4)
        else:
            x = self.base_resnet.base.layer1(x)
            fc_1 = self.classifier1(x.permute(0, 2, 3, 1))
            fc_1 = self.classifier(fc_1)
            x, cam_1 = self.genA2B(x, fc_1, self.training)
            x = self.SEM4(x, cam_1)
            
            x = self.base_resnet.base.layer2(x)
            fc_2 = self.classifier2(x.permute(0, 2, 3, 1))
            fc_2 = self.classifier(fc_2)
            x, cam_2 = self.genA2B(x, fc_2, self.training)
            x = self.SEM4(x, cam_2)
            
            real_layer3 = self.base_resnet.base.layer3(x)
            fc_3 = self.classifier3(real_layer3.permute(0, 2, 3, 1))
            fc_3 = self.classifier(fc_3)
            x, cam_3 = self.genA2B(real_layer3, fc_3, self.training)
            x = self.SEM4(x, cam_3)
            
            x = self.base_resnet.base.layer4(x)
            fc_4 = self.classifier4(x.permute(0, 2, 3, 1))
            fc_4 = self.classifier(fc_4)
            x, cam_4 = self.genA2B(x, fc_4, self.training)
            x = self.SEM4(x, cam_4)

        original_x = x
        p1_, r1, r2 = self.cbam(x)
        r1_norm = r1 / (r1.abs().mean() + 1e-8)
        r2_norm = r2 / (r2.abs().mean() + 1e-8)
        p1_enhanced = p1_ + self.idfr_residual_weight * (r1_norm + r2_norm)
        feat_idfr = p1_enhanced + original_x
        feat_idfr = self.maxpool1(feat_idfr)

        feat_enhi = None
        if real_layer3 is not None:
            try:
                adjusted_layer3 = self.layer3_adjust(real_layer3)
                p2_branch1 = self.p2_0(adjusted_layer3)
                p2_branch2 = self.p2_1(p2_branch1)
                p2_branch3 = self.p2_2(p2_branch2)
                b, c, h_orig, w_orig = p2_branch1.size()
                p2_branch1_gap = F.adaptive_avg_pool2d(p2_branch1, (1, 1)).view(b, c)
                p2_branch2_gap = F.adaptive_avg_pool2d(p2_branch2, (1, 1)).view(b, c)
                p2_branch3_gap = F.adaptive_avg_pool2d(p2_branch3, (1, 1)).view(b, c)
                feat_enhi = (p2_branch1_gap + p2_branch2_gap + p2_branch3_gap) / 3.0
                feat_enhi = F.normalize(feat_enhi, p=2, dim=1)
            except Exception as e:
                print(f"Warning: EnHi fusion failed: {e}")
                feat_enhi = None
        
        # PSMM（VSP）
        feat_vsp, sim_vsp, part_feat_vsp = self.VSP(x)

        if self.gm_pool == 'on':
            b, c, h, w = p1_.shape
            x_pool = (torch.mean(p1_.view(b, c, -1)**3, dim=-1) + 1e-12)**(1/3)
        else:
            x_pool = self.avgpool(p1_)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        if feat_enhi is not None:
            if feat_enhi.size(1) == x_pool.size(1):
                x_pool = x_pool + self.enhi_weight * feat_enhi

        x_pool = x_pool + self.vsp_weight * feat_vsp
        
        feat = self.bottleneck(x_pool)
        
        if self.training:
            out0 = self.classifier(feat)
            return x_pool, out0, x_fc, sim_vsp, part_feat_vsp
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class VisibleModule(nn.Module):
    def __init__(self, arch='resnet50'):
        super(VisibleModule, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x, x01):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        b, c, h, w = x.shape
        mask1 = torch.zeros(b, 1, h, w).to(x.device)
        x1 = torch.cat([x, mask1], dim=1)
        return x, x1


class ThermalModule(nn.Module):
    def __init__(self, arch='resnet50'):
        super(ThermalModule, self).__init__()
        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t
        self.kernel = torch.nn.Parameter(torch.ones(1, 1, 3, 3), requires_grad=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, x01):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x01 = 0.114*x01[:,:,:,0:1] + 0.587*x01[:,:,:,1:2] + 0.299*x01[:,:,:,2:3]
        x01 = x01.permute(0, 3, 1, 2)
        mask = (x01 > 220).float()
        filled = F.conv2d(mask, self.kernel, padding=1) > 8.0
        filled = F.conv2d(filled.float(), self.kernel, padding=1) < 1.0
        mask = (filled.squeeze(1) == 0).float().unsqueeze(1)
        mask = self.maxpool(self.maxpool(mask))
        x1 = torch.cat([x, mask], dim=1)
        return x, x1


class BaseResNet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(BaseResNet, self).__init__()
        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            nn.init.zeros_(m.bias.data)