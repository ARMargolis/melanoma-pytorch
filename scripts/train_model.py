import os, sys
from os import path
sys.path.append
sys.path.append(os.path.abspath('..'))
import functools
from datetime import datetime
#print(sys.path)
#import ipdb;ipdb.set_trace()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from torch.optim import Adam

from model.model import MelanomaNet
from model.data import MelanomaDataset


MODEL_LEN = 26 #20
LR = 0.005
EPOCHS = 32
BATCH_SIZE = 3
N_WORKERS=1





if __name__ == "__main__":
    train_imgs_dir = path.normpath(path.join(os.getcwd(),'../data/jpeg/train'))
    test_imgs_dir = path.normpath(path.join(os.getcwd(),'../data/jpeg/train')) # for the moment; normally ...jpeg/test
    train_csv_path = path.normpath(path.join(os.getcwd(),'../data/labels/sandbox.csv'))
    test_csv_path = path.normpath(path.join(os.getcwd(),'../data/labels/sandbox.csv')) # Just for the moment

    train_dataset = MelanomaDataset(imgs_dir=train_imgs_dir, label_csv = train_csv_path, train = True)
    test_dataset = MelanomaDataset(imgs_dir=test_imgs_dir, label_csv = test_csv_path, train = False)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True,num_workers=N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    model_config = {
    'input_shape':(1,3,244,244),
    'n_classes':1, #binary classification
    'base_channels':3,
    'block_type':'basic',
    'depth':MODEL_LEN, # depth should be two more than a multiple of six
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}

    model = MelanomaNet(config = model_config)
    a = input('Is this ok? (y/n)')
    if a == 'n':
        sys.exit(1)

    # Make sure everything is on the same device
    assert model.device == train_dataset.device

    optimizer = Adam(model.parameters(), lr=0.005)
    train_losses = {}
    test_losses = {}
    accuracies = {}

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for i, batch in enumerate(train_loader):
            #zero the optimizer
            optimizer.zero_grad()

            x, y = batch
            #with torch.no_grad():
            #    model.eval()
            #    print(model.predict(x))
            #    model.train()
            a = model(x)
            a = sigmoid(a)
            loss = (a-y)**2
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        train_losses[epoch] = np.mean(np.array(losses))

        if epoch%10 == 0:
            # eval time
            model.eval()
            with torch.no_grad():
                pred = []
                labels = []
                temp_test_losses = []
                for i, batch in enumerate(test_loader):

                    x, y = batch
                    #with torch.no_grad():
                    #    model.eval()
                    #    print(model.predict(x))
                    #    model.train()
                    a = model(x)
                    a = sigmoid(a)
                    loss = (a-y)**2
                    loss = loss.mean()
                    temp_test_losses.append(loss.item())
                    # Append binary predictions
                    pred.append((a>0.5).cpu().numpy().astype(int))
                    labels.append(y.cpu().numpy().astype(int))

                pred = np.concatenate(pred) # all predictions
                labels = np.concatenate(labels) # all labels
                acc = np.mean((pred == labels))
                accuracies[epoch] = acc
                test_loss = np.mean(np.array(temp_test_losses))
                test_losses[epoch] = test_loss
                print('Epoch {}|| Train Loss: {}, Test Loss: {}, Acc: {}'.format(epoch,train_losses[epoch],test_losses[epoch],accuracies[epoch]))
    
    # Collect metrics, generate plots
    metrics_df = pd.DataFrame([train_losses, test_losses, accuracies]).T
    metrics_df.columns = ['train', 'test', 'acc']
    metrics_df = metrics_df.interpolate(method='linear', axis = 0)
    
    #save the plots and the model
    time = str(datetime.now())
    savedir =  path.join(os.path.abspath('../data/saved_models'), time)
    os.mkdir(savedir)
    loss_plot = metrics_df[['train','test']].plot(kind='line').get_figure();loss_plot.savefig(path.join(savedir,'losses.jpg'))
    plt.clf()
    accuracy_plot = metrics_df['acc'].plot(kind='line').get_figure();accuracy_plot.savefig(path.join(savedir,'accuracy.jpg'))
    
    torch.save({'config':model_config,'state_dict':model.state_dict()}, path.join(savedir,str(EPOCHS)+'_epochs.pt') )
