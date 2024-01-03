import torch
import os

def make_model(args, parent=False):
    return Morbo()

class FreezeChannel(torch.nn.Module):
   def __init__(self, channel_idx):
       super(FreezeChannel, self).__init__()
       self.channel_idx = channel_idx


   def backward(self, module):

        #print(module.shape)
        if module.shape == torch.Size([32, 4, 3, 3]):
            module[:, self.channel_idx :, : , :] = 0
            

        elif module.shape == torch.Size([16, 32, 3, 3]):
            module[self.channel_idx: ,:, : , :] = 0

        return module
     

class RDNN_conv(torch.nn.Conv2d):
    def __init__(self, first_conv, last_conv, in_channels, out_channels, kernel_size, stride, padding, name):
        super(RDNN_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.first_conv = first_conv
        self.last_conv = last_conv  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.name = name

    def parameter_override(self, x, first_conv, last_conv):

        #rank = os.environ["OMPI_COMM_WORLD_RANK"]
        rank = 0
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        if first_conv == True:
            weights_split = torch.split(self.weight, (3, 1), dim=1)
            zeros = torch.zeros(weights_split[0].shape[0], 1, weights_split[0].shape[2], weights_split[0].shape[3]).to(device)
            weights_updated = torch.cat((weights_split[0], zeros), dim=1)

            return weights_updated
            
        
        else:
            weights_split = torch.split(self.weight, (12, 4), dim=0)
            zeros = torch.zeros(4, weights_split[0].shape[1], weights_split[0].shape[2], weights_split[0].shape[3]).to(device)
            weights_updated = torch.cat((weights_split[0], zeros), dim=0)

            bias_split = torch.split(self.bias, (12, 4), dim=0)
            ones = 1.0 - torch.zeros(4).to(device)

            bias_updated = torch.cat((bias_split[0], ones), dim=0)

            return weights_updated, bias_updated
        

    def forward(self, x):
        
        if self.first_conv:
            new_weight = self.parameter_override( x, self.first_conv, self.last_conv)
            self.weight.data = new_weight
            out = self._conv_forward(x, self.weight, self.bias)

        else:
            new_weight, new_bias = self.parameter_override( x, self.first_conv, self.last_conv)
            self.weight.data = new_weight
            self.bias.data = new_bias
            out = self._conv_forward(x, self.weight, self.bias)
        return out


class ResidualDenseBlock_4C(torch.nn.Module):
    def __init__(self, gc=32, bias=True):
        super(ResidualDenseBlock_4C, self).__init__()
        nf = gc
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=bias)
        self.relu = torch.nn.ReLU( inplace=True)

        

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x


class Morbo(torch.nn.Module):
    def __init__(self, gc=32):
        super(Morbo, self).__init__()
        #self.conv1 = RDNN_conv(in_channels = 4, out_channels = gc, kernel_size=(3,3), stride=(2,2), padding=(1,0), first_conv = True, last_conv = False, name = 'conv1' )
        self.conv1 = RDNN_conv(in_channels = 4, out_channels = gc, kernel_size=(3,3), stride=(2,2), padding=(1,1), first_conv = True, last_conv = False, name = 'conv1' )
        self.RDB1 = ResidualDenseBlock_4C(gc)
        self.RDB2 = ResidualDenseBlock_4C(gc)
        self.conv_final = RDNN_conv(in_channels= gc, out_channels = 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), first_conv = False, last_conv = True, name ='conv_final')
        self.pixel_shuffle = torch.nn.PixelShuffle(2)

        self.freeze_channel1 = FreezeChannel(channel_idx=3)
        self.freeze_channel2 = FreezeChannel(channel_idx=12)

        # Register a backward hook for the first and the last convolutions
        self.conv1.weight.register_hook(self.freeze_channel1.backward)
        self.conv_final.weight.register_hook(self.freeze_channel2.backward)


    
    def forward(self, x):
      
        x1 = self.conv1(x)
        
        out = self.RDB1(x1)
        out = self.RDB2(out)
        out = out * 0.2 + x1

        final_x = self.conv_final(out)
      
        final_x = self.pixel_shuffle(final_x)
        return final_x


def calculate_parameters(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_shape = param.shape
        total_parameters += param_shape.numel()
        print(f"{name}: {param_shape} {param_shape.numel()}")
    print(f"Total Parameters: {total_parameters}")


if __name__ == "__main__":
    model = Morbo()
    calculate_parameters(model)
    
    