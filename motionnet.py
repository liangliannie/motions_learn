import torch
import numpy as np

class UNet_down_block(torch.nn.Module):

    def __init__(self, input_channel, output_channel, down_sample):
        super(UNet_down_block, self).__init__()
        kernel_size = 3
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.down_sampling = torch.nn.Conv3d(input_channel, input_channel, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.down_sample = down_sample


    def forward(self, x):
        if self.down_sample:
            x = self.down_sampling(x)
        x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        return x

class UNet_up_block(torch.nn.Module):

    def __init__(self, prev_channel, input_channel, output_channel, ID):
        super(UNet_up_block, self).__init__()
        kernel_size = 3
        self.ID = ID
        self.up_sampling = torch.nn.ConvTranspose3d(input_channel, input_channel, 4, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias= False)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias= False)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1), bias= False)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)


    def forward(self, prev_feature_map, x):

        if self.ID == 1:
            x = self.up_sampling(x)
        # elif self.ID == 2:
        #     x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        # elif self.ID == 3:
        #     x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='area') #‘nearest’ | ‘linear’ | ‘bilinear’ | ‘trilinear’ | ‘area’
        # print(x.shape, prev_feature_map.shape)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        return x


class UNet(torch.nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # self.opts = opts
        input_channel_number = 1
        output_channel_number = 1
        # kernel_size = opts.kernel_size # we could change this later
        kernel_size = 3
        # Encoder network
        self.down_block1 = UNet_down_block(input_channel_number, 2, False) # 64*520
        self.down_block2 = UNet_down_block(2, 4, True) # 64*520
        self.down_block3 = UNet_down_block(4, 4, True) # 64*260


        # bottom convolution
        self.mid_conv1 = torch.nn.Conv3d(4, 4, kernel_size, padding=(1, 1, 1), bias=False)# 64*260
        self.bn1 = torch.nn.BatchNorm3d(4)
        self.mid_conv2 = torch.nn.Conv3d(4, 4, kernel_size, padding=(1, 1, 1), bias=False)# 64*260
        self.bn2 = torch.nn.BatchNorm3d(4)
        self.mid_conv3 = torch.nn.Conv3d(4, 4, kernel_size, padding=(1, 1, 1), bias=False) #, dilation=4 # 64*260
        self.bn3 = torch.nn.BatchNorm3d(4)
        self.mid_conv4 = torch.nn.Conv3d(4, 4, kernel_size, padding=(1, 1, 1), bias=False)# 64*260
        self.bn4 = torch.nn.BatchNorm3d(4)
        self.mid_conv5 = torch.nn.Conv3d(4,4, kernel_size, padding=(1, 1, 1), bias=False)# 64*260
        self.bn5 = torch.nn.BatchNorm3d(4)

        # Decoder network
        self.up_block2 = UNet_up_block(4, 4, 4, 1)# 64*520
        self.up_block3 = UNet_up_block(2, 4, 2, 1)# 64*520
        # Final output
        self.last_conv1 = torch.nn.Conv3d(2, 2, 3, padding=(1, 1, 1), bias=False)# 64*520
        self.last_bn = torch.nn.BatchNorm3d(2) # 64*520
        self.last_conv2 = torch.nn.Conv3d(2, 1, 3, padding=(1, 1, 1))# 64*520
        # self.linear1 = torch.nn.Sequential(*self.lin_tan_drop(64*520, 1024))
        # self.linear2 = torch.nn.Sequential(*self.lin_tan_drop(1024, 64*520))
        self.softplus = torch.nn.Softplus(beta=5, threshold=100)
        # self.softplus = torch.nn.ReLU()

    # def lin_tan_drop(self, num_features_in, num_features_out, dropout=0.5):
    #     layers = []
    #     layers.append(torch.nn.Linear(num_features_in, num_features_out, bias=True))
    #     layers.append(torch.nn.Tanh())
    #     layers.append(torch.nn.Dropout(p=dropout))
    #     return layers

    def forward(self, x, test=False):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)

        x4 = torch.nn.functional.leaky_relu(self.bn1(self.mid_conv1(x3)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn2(self.mid_conv2(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn3(self.mid_conv3(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.mid_conv4(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn5(self.mid_conv5(x4)), 0.2)

        out = self.up_block2(x2, x4)
        out = self.up_block3(x1, out)
        out = torch.nn.functional.relu(self.last_bn(self.last_conv1(out)))
        out = self.last_conv2(out)
        # change the output
        out = (out + x)
        # out = torch.sigmoid(out) #Try tanh and scale
        # out = torch.nn.functional.relu(self.last_bn2(self.last_conv2(out)))
        # out = torch.nn.functional.relu(self.last_conv2(out))
        # out = self.linear1(out.reshape(self.opts.batch_size, -1))
        # out = self.linear2(out).reshape(self.opts.batch_size, 1, 64, 520)
        # out = torch.nn.functional.sigmoid(out)
        # out = self.softplus(out)
        out = torch.sigmoid(out)
        print(out.shape)
        return out


if __name__ == '__main__':

    net = UNet()
    x = torch.randn(1, 1, 159, 220, 220)
    x = np.pad(x, ((0, 0), (0, 0), (1, 0), (0, 0),(0, 0)),'mean')
    x = torch.tensor(x)
    out = net(x)
    print(out.shape)
    #
    # # x = torch.autograd.Variable(torch.rand(1, 3, 256, 256))
    # out = framelets(x)
    #
    # print(out.shape)
    # print(out)


    # Load image
    import numpy as np
    import matplotlib.pylab as plt