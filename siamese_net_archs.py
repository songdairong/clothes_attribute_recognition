from torch import nn
import torch
import torch.nn.functional as F


class SiameseNetSimpleDoubleOut(nn.Module):
    # https://github.com/delijati/pytorch-siamese/blob/master/train_mnist.py
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNet5LayersDoubleOut2(nn.Module):
    # https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(2304, 2)


    def forward_once(self, x):
        # Forward pass

        output = self.conv1(x)
        output = F.relu(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = F.relu(output)
        output = self.conv3(output)
        output = F.relu(output)
        output = output.view(output.shape[0], -1)
        output = self.linear1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
    


# https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/
class SiameseNetworkCustom(nn.Module):
    def __init__(self):
        super().__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(65536, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 2),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetworkPretrainedBackboneSigmoid2(nn.Module):
    def __init__(self, pretrained_resnet):
        super().__init__()
        self.backbone = pretrained_resnet
        backbone_fc_outputs = pretrained_resnet.fc.out_features
        self.linear2 = nn.Linear(backbone_fc_outputs, 1)

    def forward(self, data):
        res = []
        for i in range(2):
            output = self.backbone(data[i])
            output = output.view(output.size()[0], -1)
            output = F.relu(output)
            res.append(output)
        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        output = torch.sigmoid(res)
        return output
    
    
class SiameseNetworkPretrainedBackbone(nn.Module):
    def __init__(self, pretrained_resnet, out_act, out_neurons):
        super().__init__()
        assert out_act in ['no_act', 'sigmoid'], "select out act from ['no_act', 'sigmoid']"
        self.out_act = out_act
        self.backbone = pretrained_resnet
        backbone_fc_outputs = pretrained_resnet.fc.out_features
        self.linear2 = nn.Linear(backbone_fc_outputs, out_neurons)

    def forward_once(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = F.relu(output)
        return output

    def forward(self, input1, input2):
        
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        output_res = torch.abs(output1 - output2)
        output_res = self.linear2(output_res)
        if self.out_act=="sigmoid":
            output_res = torch.sigmoid(output_res)
        return output_res
        
        

