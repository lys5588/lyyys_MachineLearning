import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import CNNDLGA

import pickle

def get_loader(dataset, batch_size=100, shuffle=True, num_workers=2):
    """Builds and returns Dataloader."""


    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
    return data_loader

def d_attn(train_data,valid_data,test_data,D_all):
    # Hyper Parameters
    input_size = 10000
    num_epochs = 10
    batch_size = 100
    learning_rate = 1e-4

    X_user=D_all["X_user"]
    X_item=D_all["X_item"]


    train_loader=get_loader(dataset=train_data,batch_size=batch_size)
    valid_loader = get_loader(dataset=valid_data, batch_size=batch_size)
    test_loader = get_loader(dataset=test_data, batch_size=batch_size)
    print("train/val/test/: {:d}/{:d}/{:d}".format(len(train_loader), len(valid_loader), len(test_loader)))
    print("==================================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnndlga = CNNDLGA(input_size,device=device)

    if torch.cuda.is_available():
        cnndlga.cuda()

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnndlga.parameters(), lr=learning_rate)


    print("==================================================================================")
    print("Training Start..")

    batch_loss = 0
    # Train the Model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (user, item, labels) in enumerate(train_loader):

            # Convert torch tensor to Variable
            # batch_size = len(user)
            user_seq=[]
            for id in user.numpy():
                user_seq.append(X_user[id])
            user=torch.Tensor(np.array(user_seq))
            item_seq = []
            for id in item.numpy():
                item_seq.append(X_item[id])
            item=torch.Tensor(np.array(item_seq))

            if torch.cuda.is_available():
                user=user.to(device)
                item=item.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = cnndlga(user, item)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.data[0]
            if i % 10 == 0:
                # Print log info
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, num_epochs, i, total_step,
                         batch_loss / 10, np.exp(loss.data[0])))
                batch_loss = 0

        # Save the Model
        torch.save(cnndlga.state_dict(), 'model_' + str(epoch) + '.pkl')

    print("==================================================================================")
    print("Training End..")

    print("==================================================================================")
    print("Testing Start..")

    for i, (user, item, labels) in enumerate(valid_loader):

        # Convert torch tensor to Variable
        batch_size = len(user)
        user_seq = []
        for id in user.numpy():
            user_seq.append(X_user[id])
        user = torch.Tensor(user_seq)
        item_seq = []
        for id in item.numpy():
            item_seq.append(X_item[id])
        item = torch.Tensor(item_seq)

        outputs = cnndlga(user, item)
        if i == 0:
            result = outputs.data.cpu().numpy()
        else:
            print
            len(result)
            result = np.append(result, outputs.data.cpu().numpy(), axis=0)

    pickle.dump(result, open('result_val.pickle', 'wb'))

    print("==================================================================================")
    print("Testing End..")
