import sys
import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

# To avoid CUDA oom (really don't know why this will happen)
torch.cuda.empty_cache()


if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    print('If use cuda: ', use_cuda)

    # Save acc and loss during both training and validation through each epoch 
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []

    # Run any number of epochs you want
    # ep = 10
    ep = 25
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            # total_loss += loss.item()
            total_loss += float(loss)
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        training_loss.append(total_loss / total_cnt)
        training_acc.append(correct_cnt / total_cnt)

        ################
        ## Validation ##
        ################
        model.eval()    # Switch to the evaluation mode --> inference mode 
        # TODO evaluation using validation set 
        valid_correct = 0 
        valid_total = 0 
        valid_loss = 0
        for data in val_loader: 
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicts = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicts == labels).sum().item()
            valid_loss += float(criterion(outputs, labels))
        print('Avg loss of the network on the validation images: %f ' % (valid_loss / valid_total))
        print('Acc of the network on the validation images: %f4 %%' % (100 * valid_correct / valid_total))
        print('Validation set size: ', valid_total)

        validation_loss.append(valid_loss / valid_total) 
        validation_acc.append(valid_correct / valid_total) 

        model.train()   # Back to training mode 

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Print model structure 
    print(model.name)

    # Plot Learning Curve
    # TODO
    def draw_acc_and_loss_fig(acc_list, loss_list, label_nm='train or validate', model_nm='unknown_model'): 
        epoch_arr = np.arange(0, len(training_acc))
        plt.figure()
        plt.title('Avg loss of {0}'.format(model_nm))
        plt.plot(epoch_arr, loss_list, label='{0} loss'.format(label_nm))    
        plt.legend()

        plt.figure()
        plt.title('Acc of {0}'.format(model_nm))
        plt.plot(epoch_arr, acc_list, label='{0} acc'.format(label_nm))
        plt.legend()

    # epoch_arr = np.arange(0, len(training_acc))
    # plt.figure()
    # plt.title('Avg loss of {0}'.format(model.name()))
    # plt.plot(epoch_arr, training_loss, label='training loss')
    # plt.show()

    draw_acc_and_loss_fig(
        training_acc, training_loss, 
        'training', model.name()
    )
    draw_acc_and_loss_fig(
        validation_acc, validation_loss, 
        'validation', model.name()
    )
    plt.show()

    