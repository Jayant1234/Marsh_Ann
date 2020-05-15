import math
import torch.nn as nn


class NeatBlock(nn.Module): 

    def __init__(self,BatchNorm=nn.BatchNorm2d,inp_channel_size=1024,group_size=2,residual=False):
        super(NeatBlock, self).__init__()
        self.residual=residual
        self.first_channel=int(inp_channel_size)
        in_channel=self.first_channel
        out_channel = int(in_channel/group_size)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn2 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=int(out_channel/group_size)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn3 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=int(out_channel/group_size)
        self.conv4 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn4 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=int(out_channel/group_size)
        self.conv5 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn5 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=int(out_channel/group_size)
        self.conv6 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn6 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=int(out_channel/group_size)
        self.conv7 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=out_channel, kernel_size=3,padding=1, bias=False)
        self.bn7 = BatchNorm(out_channel)
        in_channel=out_channel
        out_channel=self.first_channel
        self.conv8 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=1, kernel_size=1,padding=0, bias=False)
        self.bn8 = BatchNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        #self.first_channel=out_channel
        #512x512xout_channel image here exactly same as input. 

    def forward(self, x):
        if (self.residual): 
            residual = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
		
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
		
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
		
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
		
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
		
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
		
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        if(self.residual): 
            x+=residual
        return x


class NeatCNN(nn.Module):
    def __init__(self,num_classes=10,BatchNorm=nn.BatchNorm2d, channel_size=1024,group_size=2,depth=1,width=1,residual=False):
        super(NeatCNN, self).__init__()
        if channel_size/256 < 1 or channel_size%128!=0: 
            assert "Channel size is INVALID"
        if channel_size/group_size < 16: 
            assert "Group_size is INVALID"
        out_channel=int(channel_size)
        #512x512x3 enter
        self.conv1 = nn.Conv2d(3, out_channels=out_channel, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = BatchNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        in_channel=out_channel
        out_channel = int(out_channel/group_size)
      
        self.layer1 = self._make_layer(NeatBlock,depth=depth, inp_channel_size=channel_size,group_size=group_size,residual=residual)

        self.conv8 = nn.Conv2d(in_channel, 64, groups=1, kernel_size=2, padding=0,stride=2, bias=False)
        self.bn8 = BatchNorm(64)
        #256
        self.conv9 = nn.Conv2d(64, 32, groups=1, kernel_size=2, padding=0,stride=2 ,bias=False)
        self.bn9 = BatchNorm(32)
        #128
        self.conv10 = nn.Conv2d(32, 16, groups=1, kernel_size=2, padding=0,stride=2, bias=False)
        self.bn10 = BatchNorm(16)
        #64
        self.conv11 = nn.Conv2d(16, 8, groups=1, kernel_size=2, padding=0,stride=2 , bias=False)
        self.bn11 = BatchNorm(8)
        #32*32*8 expected but why am i getting 31*31*8? ,bias=False? 
        self.classifier=nn.Linear(32*32*8, num_classes)		
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self,block,depth=1,inp_channel_size=1024,group_size=2,residual=False ):
        layers = []
        for i in range(0, depth):
            layers.append(block(inp_channel_size=inp_channel_size,group_size=group_size,residual=residual))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x =  self.layer1(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
		
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
		
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
		
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)

        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x=self.classifier(x)

        return x	

	

if __name__ == "__main__":
    import torch
	
    model = NeatCNN(num_classes=10,channel_size=256,group_size=2,depth=3,width=1,residual=True).cuda()
    input = torch.rand(1, 3, 512, 512).cuda()
    output = model(input)
    print(output.size())
    print(output)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    #print(low_level_feat.size())
