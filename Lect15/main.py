import cv2 as cv2
import os
import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Type, Dict, Any
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from tqdm import tqdm
import os.path
import imgaug as ia
import os
from threading import Thread
from queue import Empty, Queue
import threading
import random
import accimage
from libs import *
from libs.plot_examples import plot_examples, plot_examples_segmentation
from torch.utils.tensorboard import SummaryWriter
from os.path import join, isfile, isdir
from seg_losses import *



resize = 1024
tb_period = 8


run_prefix = 'Unet_CoordConv_mish_resize1024'

# region logs_basepath
existing_logs_directories = [d for d in find_directories('./logs', '%s_run*' % run_prefix, maxdepth=2)]
prev_runs = [os.path.basename(os.path.split(d)[0]) for d in existing_logs_directories]
prev_runs = [int(s.replace('%s_run' % run_prefix, '')) for s in prev_runs]
if len(prev_runs) > 0:
    curr_run = np.max(prev_runs) + 1
else:
    curr_run = 1
curr_run = '%s_run%04d' % (run_prefix, curr_run)
logs_basepath = os.path.join('./logs', curr_run)
tb_basepath = os.path.join('./TBoard', curr_run)
EnsureDirectoryExists(logs_basepath)

checkpoints_basepath = os.path.join('./checkpoints', curr_run)
EnsureDirectoryExists(checkpoints_basepath)
# endregion


# region backing up the scripts configuration
EnsureDirectoryExists('./scripts_backup')
print('backing up the scripts')
ignore_func = lambda dir, files: [f for f in files if (isfile(join(dir, f)) and f[-3:] != '.py' and f[-4:] != '.csv' and f[-6:] != '.ipynb')] + [d for d in files if ((isdir(d)) & (('scripts_backup' in d) |
                                                                                                                                        ('__pycache__' in d) |
                                                                                                                                        ('.pytest_cache' in d) |
                                                                                                                                        d.endswith('.ipynb_checkpoints') |
                                                                                                                                        d.endswith('logs.bak') |
                                                                                                                                        d.endswith('outputs') |
                                                                                                                                        d.endswith('processed_data') |
                                                                                                                                        d.endswith('build') |
                                                                                                                                        d.endswith('images') |
                                                                                                                                        d.endswith('logs') |
                                                                                                                                        d.endswith('snapshots')))]
scripts_backup_dir = os.path.join('./scripts_backup', curr_run)
copytree_multi('./', scripts_backup_dir, ignore=ignore_func)
# with open(os.path.join(scripts_backup_dir, 'launch_parameters.txt'), 'w+') as f:
#     f.writelines([f'{s}\n' for s in sys.argv])
# endregion backing up the scripts configuration

tb_writer = SummaryWriter(log_dir=tb_basepath)

device1 = torch . device ( "cuda:0" if torch . cuda . is_available () else "cpu" )
device2 = torch.device('cpu')

#region functions
def start_train(model):  # запускаем обучение всех слоев
    for param in model.parameters():
        param.requires_grad = True


def calculate_loss(model_result: torch.tensor,
                   data_target: torch.tensor,
                   loss_function: torch.nn.Module = torch.nn.MSELoss()):  # reduction = None
    lossXY = (loss_function(model_result[:, :2], data_target[:, :2])) ** (0.5)  # тут из батчей получаю, править
    # lossR = loss_function(model_result[:, 2], data_target[:, 2])  # потом корень извлечь
    #MK: сейчас нет R
    # return {lossXY.item(), lossR.item()}
    return lossXY.item()
#endregion


model = UnetCoordConv(activation = Mish)
for param in model.parameters():
    param.requires_grad = True
model = nn.DataParallel(model)
model = model.to(device1)

#region threading
class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate"""

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while tokill() == False:
        for img, mask, landmarks in dataset_generator:
            batches_queue.put((img, mask, landmarks), block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    while tokill() == False:
        (img, mask, landmarks) = batches_queue.get(block=True)
        img = torch.from_numpy(img)

        # torch normalize

        landmarks = torch.from_numpy(landmarks)
        img = Variable(img.float()).to(device1)
        landmarks = Variable(landmarks.float()).to(device1)
        cuda_batches_queue.put((img, landmarks), block=True)

        if tokill() == True:
            return
#endregion


batch_size = 4
sun_dataset_train = SunDiskDataset(csv_file='sun_disk_pos_database01train.csv',
                                   root_dir='./images',
                                   resize=resize,
                                   batch_size=batch_size)
sun_dataset_test = SunDiskDataset(csv_file='sun_disk_pos_database01test.csv',
                                  root_dir='./images',
                                  resize=resize,
                                  batch_size=batch_size,
                                  augment=False)


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function: torch.nn.Module,
                       cuda_batches_queue: Queue,
                       Per_Step_Epoch: int,
                       current_epoch: int):
    model.train()
    loss_values = []
    loss_tb = []
    pbar = tqdm(total=Per_Step_Epoch)
    for batch_idx in range(int(Per_Step_Epoch)):  # тут продумать
        data_image, target = cuda_batches_queue.get(block=True)

        ###### DEBUG plots
        if batch_idx == 0:
            plot_examples_segmentation(data_image, target, file_output = os.path.join(logs_basepath, 'train_input_ep%04d.png' % current_epoch))
        ###### DEBUG plots

        target = Variable(target)

        # target = target.to(device1)
        optimizer.zero_grad()  # обнулили\перезапустии градиенты для обратного распространения
        data_out = model(data_image)  # применили модель к данным
        loss = loss_function(data_out, target)  # применили фуннкцию потерь
        loss_values.append(loss.item())

        loss_tb.append(loss.item())
        tb_writer.add_scalar('train_loss', np.mean(loss_tb), current_epoch*Per_Step_Epoch + batch_idx)
        loss_tb=[]

        loss.backward()  # пошли по графу нейросетки обратно
        optimizer.step()  # выполняем наш градиентный спуск по вычисленным шагам в предыдущей строчке
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})
    pbar.close()
    #MK не забывай закрывать pbar

    return np.mean(loss_values)


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          Per_Step_Epoch: int,
                          current_epoch: int):
    model.eval()

    loss_values = []

    pbar = tqdm(total=Per_Step_Epoch)
    for batch_idx in range(int(Per_Step_Epoch)):  # тут продумать
        data_image, target = cuda_batches_queue.get(block=True)
        data_out = model(data_image)

        if batch_idx == 0:
            plot_examples_segmentation(data_image, data_out, file_output = os.path.join(logs_basepath, 'val_results_ep%04d.png' % current_epoch))

        loss = loss_function(data_out, target)
        loss_values.append(loss.item())
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})
    pbar.close()
    # MK не забывай закрывать pbar

    return np.mean(loss_values)


def train_model(model: torch.nn.Module,
                train_dataset: Dataset,
                val_dataset: Dataset,
                max_epochs=480):

    loss_function = BinaryFocalLoss(alpha=1, gamma=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32, T_mult=2, eta_min=1e-8, lr_decay = 0.8)

    #region data preprocessing threads starting
    batches_queue_length = 4
    preprocess_workers = 4

    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=4)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(train_thread_killer, train_batches_queue, train_dataset))
        thr.start()

    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches, args=(train_cuda_transfers_thread_killer, train_cuda_batches_queue, train_batches_queue))
    train_cudathread.start()

    test_batches_queue = Queue(maxsize=batches_queue_length)
    test_cuda_batches_queue = Queue(maxsize=4)
    test_thread_killer = thread_killer()
    test_thread_killer.set_tokill(False)

    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(test_thread_killer, test_batches_queue, sun_dataset_test))
        thr.start()

    test_cuda_transfers_thread_killer = thread_killer()
    test_cuda_transfers_thread_killer.set_tokill(False)
    test_cudathread = Thread(target=threaded_cuda_batches,
                             args=(test_cuda_transfers_thread_killer, test_cuda_batches_queue,
                                   test_batches_queue))
    test_cudathread.start()
    #endregion

    Steps_Per_Epoch_Train = 64
    Steps_Per_Epoch_Test = len(val_dataset) // batch_size + 1

    for epoch in range(max_epochs):

        print(f'Epoch {epoch} / {max_epochs}')
        train_loss = train_single_epoch(model, optimizer, loss_function, train_cuda_batches_queue, Steps_Per_Epoch_Train, current_epoch=epoch)

        tb_writer.add_scalar('train_loss', train_loss, epoch)

        val_loss = validate_single_epoch(model, loss_function, test_cuda_batches_queue, Steps_Per_Epoch_Test, current_epoch=epoch)
        tb_writer.add_scalar('val_loss', val_loss, epoch)

        print(f'Validation loss: {val_loss}')

        lr_scheduler.step()

        torch.save(model.module.state_dict(), os.path.join(checkpoints_basepath, 'model_ep%04d.pt' % epoch))

    #region stopping datapreprocessing threads
    test_thread_killer.set_tokill(True)  # убиваю потокои, так же убить валидационные
    train_thread_killer.set_tokill(True)  # убиваю потокои, так же убить валидационные
    test_cuda_transfers_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            test_batches_queue.get(block=True, timeout=1)
            test_cuda_batches_queue.get(block=True, timeout=1)
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
    #endregion


train_model(model,
            train_dataset = sun_dataset_train,
            val_dataset =  sun_dataset_test,
            max_epochs=480)