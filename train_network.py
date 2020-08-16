import numpy
import h5py
from utils import *
from dataset import *
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ResNet import *
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_DIRECTORY = None
SAVE_DIRECTORY = "models/nn.pt"

if __name__ == "__main__":

    print(DEVICE)
    print(BATCH_SIZE)

    with h5py.File("training_data/lichess_database.h5", 'r') as hf:
        states = hf["States"][:]
        policy = hf["Policy"][:]
        value = hf["Value"][:]

    data = TrainingSet(states, policy, value)

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    # this is a residual network
    model = ResNetDoubleHeadSmall().double().to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    try:
        checkpoint = torch.load(LOAD_DIRECTORY)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        totalLoss = checkpoint['loss']

    except:
        print("Pretrained NN model not found")

    policy_crit = nn.PoissonNLLLoss()
    value_crit = nn.MSELoss()

    total_step = len(trainLoader)

    for epoch in range(EPOCHS):
        for i, (images, policy_labels, value_labels) in enumerate(trainLoader):
            images = images.to(DEVICE)
            policy_labels = policy_labels.to(DEVICE)
            value_labels = value_labels.to(DEVICE).double()


            optimizer.zero_grad()

            output_policy, output_value = model(images)
            policy_loss = policy_crit(output_policy, policy_labels) * 100
            value_loss = value_crit(output_value, value_labels)
            total_loss = policy_loss + value_loss

            total_loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Policy Loss: {:.4f}, Value Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, policy_loss.item(), value_loss.item()))
            if (i + 1) % 40 == 0:
                # Save Model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, SAVE_DIRECTORY)

            if (i + 1) % 20 == 0:   # check train accuracy on a small portion of the train set.
                # find predicted labels
                values = np.exp((model(images)[0].data.detach().to(DEVICE).numpy()))
                print("MAX:", np.amax(np.amax(values, axis=1)))
                print("MIN:", np.amin(np.amin(values, axis=1)))

                _, predicted = torch.max(model(images)[0].data, 1)
                predicted = predicted.to(DEVICE).numpy()

                _, actual = torch.max(policy_labels.data, 1)  # for poisson nll loss
                actual = actual.numpy()

                print("Predicted:", predicted)
                print("Actual:", actual)

                print("Correct:", (predicted == actual).sum())
