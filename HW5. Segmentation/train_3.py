import torch
# from torch import nn
# import torch.utils.data as dt
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from tqdm import *
import numpy as np

log = './log/'
train = './data/train/'
train_masks = './data/train_masks/'
test = './data/test/'
test_masks = './data/test_masks'

import sys

sys.path.append('gdrive/My Drive/New_hw_5_NN_test/')
import carvana_dataset as cv

PATH = 'gdrive/My Drive/Technotrack_hw_5/data'
log = './log/'
train = PATH + '/train/'
train_masks = PATH + '/train_masks/'
test = PATH + '/test/'
test_masks = PATH + '/test_masks'

if os.path.exists(log) == False:
    os.mkdir(log)
tb_writer = SummaryWriter(log_dir='log')


def train_net(m, my_net_lr=0.001, useCuda=True, n_epoch=100, batch_sz=32,
              load_state_dic=True):
    """
    Делаем критерий, который будем оптимайзить
    """
    if load_state_dic:
        m.load_state_dict(torch.load('/content/gdrive/My Drive/damn_segm_model_3'))

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(m.parameters(), lr=my_net_lr)

    if useCuda == True:
        m = m.cuda()
        criterion = criterion.cuda()

    # ds = cv.CarvanaDataset(train, train_masks)
    #     ds_test = cv.CarvanaDataset(test, test_masks)

    #     dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=5)
    #     dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=5)
    dl = cv.CarvanaDataset(train, train_masks)
    dl.make_batch(int(len(dl) / batch_sz))
    dl_test = cv.CarvanaDataset(test, test_masks)
    dl_test.make_batch(int(len(dl_test) / batch_sz))

    global_iter = 0
    for epoch in range(0, n_epoch):
        print("Current epoch: ", epoch)
        epoch_loss = 0
        m.train(True)
        for iter, (i, t) in enumerate(tqdm(dl)):
            i = i.unsqueeze(0)
            t = t.unsqueeze(0)

            i = Variable(i)
            t = Variable(t)
            if useCuda:
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            loss = criterion(o, t)
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / float(len(dl))
        print("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print("Make test")
        test_loss = 0

        torch.save(m.state_dict(), '/content/gdrive/My Drive/damn_segm_model_3')

        m.train(False)

        tb_out = np.random.choice(range(0, len(dl_test)), 3)
        for iter, (i, t) in enumerate(tqdm(dl_test)):
            i = i.unsqueeze(0)
            t = t.unsqueeze(0)
            with torch.no_grad():
                i = Variable(i)
                t = Variable(t)
            if useCuda:
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            loss = criterion(o, t)
            test_loss += loss.item()

            for k, c in enumerate(tb_out):
                if c == iter:
                    tb_writer.add_image('Image/Test_input_%d' % k, i[0].cpu(), epoch)  # Tensor
                    tb_writer.add_image('Image/Test_target_%d' % k, t[0].cpu(), epoch)  # Tensor
                    tb_writer.add_image('Image/Test_output_%d' % k, o[0].cpu(), epoch)  # Tensor

        test_loss = test_loss / float(len(dl_test))
        print("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)