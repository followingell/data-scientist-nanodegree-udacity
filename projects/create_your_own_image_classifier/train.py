import argparse
import helper as hp

parser = argparse.ArgumentParser(description = 'train.py')

parser.add_argument('data_dir', nargs = '*', action = "store", default = "./flowers/", help = "folder path for data")
parser.add_argument('--gpu_cpu', dest= "gpu_cpu", action = "store", default = "gpu", help = "enable gpu computation")
parser.add_argument('--save_dir', dest = "save_dir", action = "store", default = "./checkpoint.pth", help = "filepath for saving checkpoint")
parser.add_argument('--learn_r', dest = "learn_r", action = "store", default = 0.001, help = "learning rate for the optimizer")
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5, help = "dropout value")
parser.add_argument('--epoch_num', dest = "epoch_num", action = "store", type = int, default = 3, help = "epoch value")
parser.add_argument('--architecture', dest = "architecture", action = "store", default = "vgg16", type = str, help = "specify the neural network structure: vgg16 or densenet121")
parser.add_argument('--fc2', type = int, dest = "fc2", action = "store", default = 1000, help = "state the units for fc2")

pa = parser.parse_args()
data_path = pa.data_dir
filepath = pa.save_dir
learn_r = pa.learn_r
architecture = pa.architecture
dropout = pa.dropout
fc2 = pa.fc2
gpu_cpu = pa.gpu_cpu
epoch_num = pa.epoch_num

# load the data - data_load() from help.py
trainloader, validationloader, testloader = hp.load_data(data_path)

# build model
model, optimizer, criterion = hp.nn_architecture(architecture, dropout, fc2, learn_r)

# train model
hp.train_network(model, criterion, optimizer, trainloader, validationloader, epoch_num, 20, gpu_cpu)

# checkpoint the model
hp.save_checkpoint(filepath, architecture, dropout, learn_r, fc2, epoch_num)

print("model has been successfully trained")