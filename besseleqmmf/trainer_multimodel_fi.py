import os
os.umask(0o002)

# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import h5py
import cv2
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchsummary

from ComplexNetsnoTF import *

from models import Net, NetC, NetSO2, NetTMBasis, NetTMBasis_FI


def main():
    
    parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
    parser.add_argument('--device',          dest='device',          type=str,            default='cuda',           help='cuda or cpu.')
    parser.add_argument('--model',           dest='model',           type=str,            default='mlp',            help='mlp or eq.')
    parser.add_argument('--num_workers',     dest='num_workers',     type=int,            default=0,                help='Number of workers for the dataloader.')
    parser.add_argument('--batch_size',      dest='batch_size',      type=int,            default=4,                help='Batch size for dataloader.')
    parser.add_argument('--dataset_name',    dest='dataset_name',    type=str,            default='TheoryTM_fmnist',            help='Which dataset to use. Use: TheoryTM_fmnist or TheoryTM_imagenette_grey')
    parser.add_argument('--dataset',         dest='dataset',         type=str,            default='inv',            help='Whether to use inverted images or original images as target. Use: inv or orig')
    parser.add_argument('--savedir',         dest='savedir',         type=str,            default='run0',           help='Dataset name.')
    parser.add_argument('--epochs',          dest='epochs',          type=int,            default=10,               help='Dataset name.')
    parser.add_argument('--lr1',             dest='lr1',             type=float,          default=1.0,              help='Dataset name.')
    parser.add_argument('--lr2',             dest='lr2',             type=float,          default=1.0,              help='Dataset name.')
    parser.add_argument('--max_freq',        dest='max_freq',        type=int,            default=24,                help='max_freq in SO(2) basis.')
    parser.add_argument('--num_rings',       dest='num_rings',       type=int,            default=200,              help='Dataset name.')
    parser.add_argument('--sigma',           dest='sigma',           type=float,          default=1.0,              help='Dataset name.')
    parser.add_argument('--img_size',        dest='img_size',        type=int,            default=180,              help='Dataset name.')
    parser.add_argument('--img_size_out',    dest='img_size_out',    type=int,            default=180,              help='Dataset name.')
    parser.add_argument('--loadTM',          dest='loadTM',          type=int,            default=0,                  help='Wether to load the TM model (1 or 0).')
    parser.add_argument('--loaddir',         dest='loaddir',         type=str,            default='',                help='Wether to load the TM model (1 or 0).')
    parser.add_argument('--tm_full_mat',     dest='tm_full_mat',     type=int,            default=0,                  help='Wether to use a full TM or diagonal TM.')
    parser.add_argument('--block_diag_mat',  dest='block_diag_mat',  type=int,            default=0,                  help='Trial method to create a diagonal fibre matrix with some off-diagonal elements.')
    parser.add_argument('--weight_decay',    dest='weight_decay',    type=float,          default=0,                  help='Weight decay.')
    parser.add_argument('--rand_phase',      dest='rand_phase',      type=int,            default=0,                  help='Whether to randomise the phase of spackled images.')
    args             = parser.parse_args()
    print(args)
    
    ## Load in the data
    data_path = f'data/{args.dataset_name}/'
    file_location = f'train_data.h5'
    with h5py.File( data_path + file_location , 'r') as f:
        print("Keys: %s" % f.keys())
        
        if args.dataset_name=='Real_fmnist':
            original_imgs = np.array(f['original'])
            original_imgs = original_imgs
            speckled_imgs = np.array(f['speckled'], dtype=np.single) / (255.0 * 1) 
            speckled_imgs = speckled_imgs
            
        else:
            original_imgs = np.array(f['original'])
            speckled_imgs = np.array(f['speckled'], dtype=np.csingle)
            inverted_imgs = np.array(f['inverted'], dtype=np.csingle)#[:300]
    
    data_path = f'data/{args.dataset_name}/'
    file_location = f'test_data.h5'
    with h5py.File( data_path + file_location , 'r') as f:
        print("Keys: %s" % f.keys())
        if args.dataset_name=='Real_fmnist':
            original_imgs_test = np.array(f['original'])
            speckled_imgs_test = np.array(f['speckled'], dtype=np.single) / (255.0 * 1)
        else:
            original_imgs_test = np.array(f['original'])# / 255.0
            speckled_imgs_test = np.array(f['speckled'], dtype=np.csingle)# / 255.0
            inverted_imgs_test = np.array(f['inverted'], dtype=np.csingle)
        
        
    ## Build a dataset
    class MMFDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, speckled_imgs, original_imgs, normalise_orig, rand_phase):
            """
            Args:
                original_imgs (np.array): original_imgs.
                speckled_imgs (np.array): speckled_imgs.
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.rand_phase = rand_phase
            self.speckled_imgs = speckled_imgs
            original_imgs = original_imgs.astype(np.float32)
            self.original_imgs = original_imgs

        def __len__(self):
            return len(self.original_imgs)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            speckled_imgs = self.speckled_imgs[idx]
            original_imgs = self.original_imgs[idx]
            
            if self.rand_phase == 1:
                rand_phase_offset   = np.random.rand(speckled_imgs.shape[0],speckled_imgs.shape[1]) * 2 * 3.14
                speckled_imgs_phase = np.angle(speckled_imgs)
                speckled_imgs_amp   = np.abs(speckled_imgs)
                speckled_imgs_phase = speckled_imgs_phase + rand_phase_offset
#                 speckled_imgs = np.stack([speckled_imgs_amp*np.cos(speckled_imgs_phase), 
#                                           speckled_imgs_amp*np.sin(speckled_imgs_phase)], axis=-1)    
                speckled_imgs = np.stack([speckled_imgs_amp, np.zeros((speckled_imgs_amp.shape[0],speckled_imgs_amp.shape[1]))], axis=-1)    
                speckled_imgs = channels_to_complex_np(speckled_imgs).astype(np.complex64)
        
#                 print(f'old speck img amp : {speckled_imgs_amp}')
#                 print(f'old speck img phase : {speckled_imgs_phase}')
#                 print(f'new speck img amp : {np.abs(speckled_imgs)}')
#                 print(f'new speck img phase : {np.angle(speckled_imgs)}')
            
            sample = {'speckled_imgs': speckled_imgs, 'original_imgs': original_imgs}

            return sample
    
    ## Build train/test datasets
    if args.dataset_name=='Real_fmnist':
        normalise_orig = False
    else:
        normalise_orig = True
    train_dataset = MMFDataset(speckled_imgs=speckled_imgs, original_imgs=original_imgs, normalise_orig=normalise_orig, rand_phase=args.rand_phase)
    test_dataset = MMFDataset(speckled_imgs=speckled_imgs_test, original_imgs=original_imgs_test, normalise_orig=normalise_orig, rand_phase=args.rand_phase)
    
    ## Create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    
    if args.tm_full_mat == 0:
        tm_full_mat = False
    else:
        tm_full_mat = True
    
    ## Build the model
    if args.model == 'mlp':
        mod_1 = Net(img_size1=args.img_size,img_size2=args.img_size_out)
    if args.model == 'cmlp':
        mod_1 = NetC(img_size1=args.img_size,img_size2=args.img_size_out)
    elif args.model == 'eq':
        mod_1 = NetSO2(img_size=args.img_size, max_freq=args.max_freq, num_rings=args.num_rings, sigma=args.sigma)
    elif args.model == 'tmbasis':
        if args.dataset_name=='TheoryTM_fmnist':
            mod_1 = NetTMBasis_FI(TM_name='TheoryTM_28', TM_name_in='TheoryTM', tm_full_mat=tm_full_mat, block_diag_mat=args.block_diag_mat)
        if args.dataset_name=='TheoryTM_fmnist_v3':
            mod_1 = NetTMBasis_FI(TM_name='TheoryTM_28', TM_name_in='TheoryTM', tm_full_mat=tm_full_mat, block_diag_mat=args.block_diag_mat)
        elif args.dataset_name=='TheoryTM_imagenette_grey':
            mod_1 = NetTMBasis_FI(TM_name='TheoryTM_256', tm_full_mat=tm_full_mat, block_diag_mat=args.block_diag_mat)
        elif args.dataset_name=='TheoryTM_imagenette_grey_v2':
            mod_1 = NetTMBasis_FI(TM_name='TheoryTM_256', tm_full_mat=tm_full_mat, block_diag_mat=args.block_diag_mat)
        elif args.dataset_name=='Real_fmnist':
            mod_1 = NetTMBasis_FI(TM_name='TheoryTM_MM_28', TM_name_in='TheoryTM_MM_224', Real_in=True, tm_full_mat=tm_full_mat, block_diag_mat=args.block_diag_mat)
    mod_1 = mod_1.to('cuda')
    if args.dataset_name=='Real_fmnist':
#         torchsummary.summary(mod_1, (1,args.img_size*args.img_size))
        torchsummary.summary(mod_1, (1,args.img_size_out*args.img_size_out))
    else:
#         torchsummary.summary(mod_1, (1,args.img_size,args.img_size,2))
        torchsummary.summary(mod_1, (1,args.img_size_out,args.img_size_out,2))
    
    class Conv_ReLU_Block(torch.nn.Module):
        def __init__(self):
            super(Conv_ReLU_Block, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.conv(x))
    
    class Net_SR(torch.nn.Module):
        def __init__(self, layers=6):
            super(Net_SR, self).__init__()
            self.residual_layer = self.make_layer(Conv_ReLU_Block, layers)
            self.input = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.output = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU(inplace=True)

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, sqrt(2. / n))

        def make_layer(self, block, num_of_layer):
            layers = []
            for _ in range(num_of_layer):
                layers.append(block())
            return torch.nn.Sequential(*layers)

        def forward(self, x):
            residual = x
            out = self.relu(self.input(x))
            out = self.residual_layer(out)
            out = self.output(out)
            out = torch.add(out,residual)
            return out
        
    if args.model == 'tmbasis':
#         mod_SR = Net_SR(12)
        mod_SR = Net_SR(6)
        mod_SR = mod_SR.to('cuda')
#         torchsummary.summary(mod_SR, (1,args.img_size,args.img_size))
        torchsummary.summary(mod_SR, (1,args.img_size_out,args.img_size_out))
    
    ## Optimiser, loss, create output folders
    if args.loadTM == 1:
        checkpoint = torch.load(f'outputs/{args.loaddir}/mod1_200.tar')
        mod_1.load_state_dict(checkpoint['model_1_state_dict'])
    if args.loadTM == 0:
        optimizer1 = torch.optim.SGD(mod_1.parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    if args.model == 'tmbasis':
        optimizer2 = torch.optim.SGD(mod_SR.parameters(), lr=args.lr2)
    criterion = torch.nn.MSELoss()
    outdir = f'outputs/{args.savedir}'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    if args.loadTM == 1:
        start_epoch = 201
    else:
        start_epoch = 1
        
    ## Run the training
    for epoch in range(start_epoch,args.epochs+1):
        if args.loadTM == 0:
            mod_1.train()
        if args.loadTM == 1:
            mod_1.eval()
        if args.model == 'tmbasis':
            mod_SR.train()
        tot_loss1 = 0
        tot_loss2 = 0
        tot_loss = 0
        for i, data in enumerate(trainloader):
            ## Get data for batch loop
            original_img = data['original_imgs'].to('cuda')
            speckled_img = data['speckled_imgs'].to('cuda')

            speckled_img = speckled_img.view(speckled_img.shape[0], -1)
            original_img = original_img.view(original_img.shape[0], -1)
#             original_img = torch.unsqueeze(original_img, dim=1)
            
            ## Zero optimiser gradients
#             optimizer.zero_grad()
            if args.loadTM == 0:
                optimizer1.zero_grad()
            if args.model == 'tmbasis':
                optimizer2.zero_grad()

            ## Pass image through model
#             predicted_inv_img = mod_1(speckled_img)
            predicted_speck_img, predicted_inv_img = mod_1(original_img)
            predicted_speck_img = torch.abs(predicted_speck_img)
            predicted_inv_img = torch.abs(predicted_inv_img)
            predicted_inv_img = predicted_inv_img.view(-1,1,args.img_size_out,args.img_size_out)
            if epoch > 200:
#                 for param in mod_1.parameters():
#                     param.requires_grad = False
                if args.model == 'tmbasis':
                    predicted_img = mod_SR(predicted_inv_img)
            else:
                predicted_img = predicted_inv_img

            ## Compute loss function
            predicted_img = predicted_img.view(predicted_img.shape[0], -1)
            original_img = original_img.view(original_img.shape[0], -1)
            predicted_inv_img = predicted_inv_img.view(predicted_inv_img.shape[0], -1)
            predicted_speck_img = predicted_speck_img.view(predicted_speck_img.shape[0], -1)

            loss1 = criterion(predicted_speck_img, torch.abs(speckled_img))
            if epoch > 100:
                loss2 = criterion(predicted_img, original_img)
            else:
                loss2 = 0
            
            loss = loss1
            if epoch > 100:
                loss = loss + loss2

            ## Loss backwards and step optimiser
            loss.backward()
            if args.loadTM == 0:
                optimizer1.step()
            if args.model == 'tmbasis':
                if epoch > 200:
                    optimizer2.step()

            ## Store loss
            tot_loss += loss.item()
            tot_loss1 += loss1.item()
            tot_loss2 += loss2.item()

        if args.model == 'tmbasis':
            if args.loadTM == 0:
                if (epoch % 50)==0 and epoch!=0:
                    torch.save({
                        'epoch': epoch,
                        'model_1_state_dict': mod_1.state_dict(),
                        'model_SR_state_dict': mod_SR.state_dict(),
                        'optimizer1_state_dict': optimizer1.state_dict(),
                        'optimizer2_state_dict': optimizer2.state_dict(),
                        }, f'{outdir}/mod1_{epoch}.tar')
            if args.loadTM == 1:
                if (epoch % 50)==0 and epoch!=0:
                    torch.save({
                        'epoch': epoch,
                        'model_1_state_dict': mod_1.state_dict(),
                        'model_SR_state_dict': mod_SR.state_dict(),
                        'optimizer2_state_dict': optimizer2.state_dict(),
                        }, f'{outdir}/mod1_{epoch}.tar')
        else:
            if (epoch % 50)==0 and epoch!=0:
                torch.save({
                    'epoch': epoch,
                    'model_1_state_dict': mod_1.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    }, f'{outdir}/mod1_{epoch}.tar')
            
        ## Save weights to file for TM bases model
#         with open(f'{outdir}/mod1_{epoch}_weights.txt', 'ab') as f:
#             np.savetxt(f, mod_1.cfc1.weight.detach().cpu().numpy())
#             f.write(b'\n')
            
        ## Make a test prediction
        mod_1.eval()
        if args.model == 'tmbasis':
            mod_SR.eval()
        tot_loss_test = 0
        tot_loss_test1 = 0
        tot_loss_test2 = 0
        for i, data in enumerate(testloader):
            ## Get data for batch loop
            original_img = data['original_imgs'].to('cuda')
            speckled_img = data['speckled_imgs'].to('cuda')

            speckled_img = speckled_img.view(speckled_img.shape[0], -1)
            original_img = original_img.view(original_img.shape[0], -1)
#             original_img = torch.unsqueeze(original_img, dim=1)

            ## Pass image through model
            predicted_speck_img, predicted_inv_img = mod_1(original_img)
            predicted_speck_img = torch.abs(predicted_speck_img)
            predicted_inv_img = torch.abs(predicted_inv_img)
            predicted_inv_img = predicted_inv_img.view(-1,1,args.img_size_out,args.img_size_out)
            if epoch > 200:
#                 for param in mod_1.parameters():
#                     param.requires_grad = False
                if args.model == 'tmbasis':
                    predicted_img = mod_SR(predicted_inv_img)
            else:
                predicted_img = predicted_inv_img

            ## Compute loss function
            predicted_img = predicted_img.view(predicted_img.shape[0], -1)
            original_img = original_img.view(original_img.shape[0], -1)
            predicted_inv_img = predicted_inv_img.view(predicted_inv_img.shape[0], -1)
            predicted_speck_img = predicted_speck_img.view(predicted_speck_img.shape[0], -1)

            loss1 = criterion(predicted_speck_img, torch.abs(speckled_img))
            if epoch > 100:
                loss2 = criterion(predicted_img, original_img)
            else:
                loss2 = 0
            
            loss = loss1
            if epoch > 100:
                loss = loss + loss2

            ## Store loss
            tot_loss_test += loss.item()
            tot_loss_test1 += loss1.item()
            tot_loss_test2 += loss2.item()

        print(f'epoch {epoch} - loss : {tot_loss/len(trainloader)} - test loss : {tot_loss_test/len(testloader)}')
        print(f'epoch {epoch} - loss speck : {tot_loss1/len(trainloader)} - test loss : {tot_loss_test1/len(testloader)}')
        print(f'epoch {epoch} - loss orig : {tot_loss2/len(trainloader)} - test loss : {tot_loss_test2/len(testloader)}')
    

if __name__ == "__main__":
    main()
