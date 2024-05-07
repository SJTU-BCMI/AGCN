import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]: 
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=1):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size() 
        A = self.A
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A_private=A1
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        # y += self.down(x)
        A_public=self.PA
        #A_private, A_public=[],[]
        A_private=[]
        # return self.relu(y), A_private, A_public
        return y, A_private, A_public


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1= unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(in_channels, out_channels, kernel_size=3, stride=stride)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=3, stride=stride)

    def forward(self, x):
        x_gcn, A_private, A_public =self.gcn1(x)
        # print('!!!!!!!!!!!!!!!!!x_gcn',x_gcn.size())
        # print('!!!!!!!!!!!!!!!!!self.tcn1(x)',self.tcn1(x).size())
        # print('!!!!!!!!!!!!!!!!!self.residual(x)',self.residual(x).size())
        # x=x_gcn+self.tcn1(x)+ self.residual(x)
        x=x_gcn+ self.residual(x)
        # return x_gcn, A_private, A_public
        # x = self.tcn1(x_gcn) + self.residual(x)
        # # x = x_gcn
        # # print('x')
        # #x = self.gcn1(x) + self.residual(x)
        
        return self.relu(x), A_private, A_public


class Model(nn.Module):
    def __init__(self, num_class=5, num_point=62, num_person=1, A=None, graph_args=dict(), in_channels=5,c1=64, c2=64, c3=64, c4=128, c5=128, c6=256):
        super(Model, self).__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
        #     Graph = import_class(graph)
        #     self.graph = Graph(**graph_args)

        #A = self.graph.A
      
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, c1, A, residual=False)

        self.l2 = TCN_GCN_unit(c1, c2, A)

        self.l3 = TCN_GCN_unit(c2, c3, A)

        self.l4= TCN_GCN_unit(c3, c4, A)

        self.l5 = TCN_GCN_unit(c4, c5, A)

        self.l6 = TCN_GCN_unit(c5, c6, A)

        # self.l7 = TCN_GCN_unit(c4, c5, A)
        # self.l8 = TCN_GCN_unit(c5, c6, A, stride=2)
        # self.l9 = TCN_GCN_unit(c6, c6, A)
        # self.l10 = TCN_GCN_unit(c6, c6, A)

        self.fc = nn.Linear(c3, num_class)
        # self.fc = nn.Linear(c6, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()#N：batch size;C: in_channels 5;T:time lenth 4;V:vertex(num_point=64):M: num_person 1
        #print(N, C, T, V, M)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        A_private_list=[]
        A_public_list=[]  

        x , A_private, A_public= self.l1(x)
        A_private_list.append(A_private)
        A_public_list.append(A_public)
        x , A_private, A_public= self.l2(x)
        A_private_list.append(A_private)
        A_public_list.append(A_public)
        x , A_private, A_public= self.l3(x)
        A_private_list.append(A_private)
        A_public_list.append(A_public)
        # x , A_private, A_public = self.l4(x)
        # A_private_list.append(A_private)
        # A_public_list.append(A_public)
        # x , A_private, A_public= self.l5(x)
        # A_private_list.append(A_private)
        # A_public_list.append(A_public)
        # x , A_private, A_public= self.l6(x)
        # A_private_list.append(A_private)
        # A_public_list.append(A_public)
    
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x),A_private_list, A_public_list