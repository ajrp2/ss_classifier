import argparse

import torch
from torch.utils import data

from classes import Conv3DModel, SomiteStageDataset
import conf
from conf import PARAMS as params
import utils as ut

## Command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default=None, help="Path to the directory containing the training and validation data.")
parser.add_argument("--n_epochs", type=int, default=10000, help="Number of iterations through the entire dataset.")
parser.add_argument("--real", type=int, default=None, help="Real or test run")

opt = parser.parse_args()

# SigOpt conf
if opt.real == None:
    # Dev token
    token = "LYPULIWFCQQWESHFGGAISHKRXALVXYRHBUHWDXJZCORKWKIG"
    print("Using DEV token for the SigOpt API")

else:
    # Real token
    token = "PJYAHHJQGOINNXZCWYYJIIZYBGVFGTBWYERAMUUQDXFVFOEE"    
    print("WARNING /n Using REAL token for the SigOpt API")


# CUDA for PyTorch
if torch.cuda.is_available():
    device = torch.cuda.device("cuda:0")
    torch.backends.cudnn.benchmark = True 
    print("CUDA is available")

else:
    device = torch.device("cpu")    
   

all_img_folder = ut.get_image_paths(
    conf.SOMITE_COUNTS,
    opt.data_dir
)

train_folder, val_folder = ut.test_train_split(
    opt.data_dir,
    all_img_folder,
    0.2,
    conf.N_CLASSES
)

# Datasets
train_data = SomiteStageDataset(
    img_folder=train_folder
)

val_data = SomiteStageDataset(
    img_folder=val_folder
)


# Generators
train_generator = data.DataLoader(
    train_data,
    batch_size=conf.BATCH_SIZE,
    num_workers=1,
    shuffle=True
)

val_generator = data.DataLoader(
    val_data,
    batch_size=conf.BATCH_SIZE,
    num_workers=1,
    shuffle=True
)


model = Conv3DModel().to(device)
model.train()
model.apply(ut.weights_init)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

            
# Train the model
total_step = len(train_generator)
loss_list = []
acc_list = []
num_epochs = opt.n_epochs

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_generator):

        # Transfer to CPU/GPU
        images = torch.FloatTensor(images).to(device)
        labels = labels.to(device)

        # Forward pass: Compute predicted labels by passing images to the model
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in val_generator:

        # Transfer to CPU/GPU
        images = torch.FloatTensor(images).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(str(len(val_data)), (correct / total) * 100))
