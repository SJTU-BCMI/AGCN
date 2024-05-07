# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from AGCN import Model
import random


if __name__ == "__main__":

    n_sample = 1000
    n_channel = 62
    n_band = 5
    train_data = torch.randn((n_sample,n_band,1,n_channel,1))
    train_label = torch.randint(low=0, high=2, size=(n_sample,))  
    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )
    A=np.zeros((n_channel,n_channel))
    num_class = 2
    b = 5
    c1 =random.randint(30,50)
    c2 =random.randint(40,60)
    c3 =random.randint(50,70)
    c4 =random.randint(60,80)
    c5 =random.randint(70,90)
    c6 =random.randint(80,120)
    m = Model(A=A.reshape(1,n_channel,n_channel),num_class=num_class, num_point=n_channel,in_channels=b,c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6)
    m = m
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(50):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.float()
            b_x = b_x
            b_y = b_y
            b_y = b_y.long()
            output,A_private_tr, A_public_tr = m(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            corr_step=pred.eq(b_y.data).cpu().sum()
        