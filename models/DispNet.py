import torch
import torch.nn as nn


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_disp(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class DispNet(nn.Module):

    def __init__(self, num_d = 8, batchNorm=True):
        super(DispNet,self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 3 + num_d, 64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)


        self.upconv6= upconv(1024,512)
        self.upconv5 = upconv(512,256)
        self.upconv4 = upconv(256,128)
        self.upconv3 = upconv(128,64)
        self.upconv2 = upconv(64,32)
        self.upconv1 = upconv(32,16)


        self.iconv6 = conv(self.batchNorm, 1024, 512)
        self.iconv5 = conv(self.batchNorm, 769 , 256)
        self.iconv4 = conv(self.batchNorm, 385 , 128)
        self.iconv3 = conv(self.batchNorm, 193 , 64)
        self.iconv2 = conv(self.batchNorm, 97 , 32)
        self.iconv1 = conv(self.batchNorm, 17, 16)


        self.predict_disp6 = predict_disp(512, 1)
        self.predict_disp5 = predict_disp(256, 1)
        self.predict_disp4 = predict_disp(128, 1)
        self.predict_disp3 = predict_disp(64, 1)
        self.predict_disp2 = predict_disp(32, 1)
        self.predict_disp1 = predict_disp(16, 1)



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.kaiming_normal(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_upconv6 = crop_like(self.upconv6(out_conv6), out_conv5)
        concat6 = torch.cat((out_conv5, out_upconv6), 1)
        out_iconv6 = self.iconv6(concat6)

        disp6       = self.predict_disp6(out_iconv6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        disp6_up = crop_like(nn.functional.interpolate(disp6, scale_factor=2, mode='bilinear', align_corners=False), out_conv4)
        concat5 = torch.cat((out_conv4, out_upconv5, disp6_up),1)
        out_iconv5 = self.iconv5(concat5)

        disp5       = self.predict_disp5(out_iconv5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        disp5_up    = crop_like(nn.functional.interpolate(disp5, scale_factor=2, mode='bilinear', align_corners=False), out_conv3)
        concat4 = torch.cat((out_conv3,out_upconv4,disp5_up),1)
        out_iconv4 = self.iconv4(concat4)

        disp4       = self.predict_disp4(out_iconv4)

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up    = crop_like(nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_conv2,out_upconv3,disp4_up),1)
        out_iconv3 = self.iconv3(concat3)

        disp3       = self.predict_disp3(out_iconv3)

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up    = crop_like(nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_conv1,out_upconv2,disp3_up),1)
        out_iconv2 = self.iconv2(concat2)

        disp2       = self.predict_disp2(out_iconv2)

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up    = crop_like(nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1,disp2_up),1)
        out_iconv1 = self.iconv1(concat1)

        disp1       = self.predict_disp1(out_iconv1)

        if self.training:
            return disp1, disp2, disp3, disp4, disp5, disp6
        else:
            return disp1
