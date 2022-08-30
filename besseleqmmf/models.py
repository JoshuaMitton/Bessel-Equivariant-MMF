import torch
from e2cnn import gspaces
from e2cnn import kernels
from e2cnn import nn
import pickle
from ComplexNetsnoTF import *


class Hadamard2(torch.nn.Module):
    # creates a custom layer for elementwise multiplying every element of a vector by another weight vector
    # assumes input is complex in separate real and imaginary channels
    def __init__(self, in_features, **kwargs):
        super(Hadamard2, self).__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_features))
        torch.nn.init.uniform_(self.weight,-np.sqrt(1/in_features),np.sqrt(1/in_features))

    def forward(self, x):
        if x.shape[-1] == 2:
            weight = torch.cat((torch.unsqueeze(self.weight,-1), 0*torch.unsqueeze(self.weight,-1)), -1)
            return complex_to_channels(channels_to_complex(x) * 
                                       torch.view_as_complex(weight))
        else:
            return x*self.weight
        


class Net(torch.nn.Module):
    def __init__(self,img_size1,img_size2):
        super().__init__()
        self.amp = Amplitude()
        self.fc1 = torch.nn.Linear(img_size1*img_size1, img_size2*img_size2)
        self.hadamard2 = Hadamard2(img_size2*img_size2)
        
    def forward(self, x):
        x = self.amp(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.hadamard2(x)
        return x
    
class NetC(torch.nn.Module):
    def __init__(self, img_size1,img_size2):
        super().__init__()
        self.fc1 = ComplexDense(img_size1*img_size1, img_size2*img_size2)
        self.hadamard2 = Hadamard2(img_size2*img_size2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.hadamard2(x)
        x = channels_to_complex(x)
        return x
    

## Bases for SO(2) equivariant dense model from e2cnn package
## https://github.com/QUVA-Lab/e2cnn

## Compute and visualise all radial bases functions for the group SO(2) on 28x28 bases
def rad_bases(max_freq=5, kernel_size=180, num_rings=None, sigma=None, plot_bases=False):
    irreps = [f'irrep_{i}' for i in range(max_freq)]

    r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=max_freq)

    if num_rings is not None:
        rings = torch.linspace(0, (kernel_size-1)//2, num_rings)
        rings = rings.tolist()
    else:
        rings = None

    print(f'''Bases parameters used to compute SO(2) equivariant bases : 
    max frequency = {max_freq}, kernel size = {kernel_size}, 
    number of rings/radii = {num_rings}, sigma value = {sigma}''')

    num_bases = 0
    bases_re = []
    bases_im = []
    for irrep_num, irrep in enumerate(irreps):

        feat_type_in = nn.FieldType(r2_act, [r2_act.irreps['irrep_0']])
        feat_type_out = nn.FieldType(r2_act, [r2_act.irreps[f'{irrep}']])

        conv = nn.R2Conv(feat_type_in,feat_type_out,kernel_size=kernel_size, rings=rings, sigma=sigma)

        bases_shape = conv._basisexpansion._modules[f"block_expansion_(\'irrep_0\', \'{irrep}\')"]._buffers['sampled_basis'].shape
        num_bases += bases_shape[0]*bases_shape[1]

#         print(bases_shape)
        for j in range(bases_shape[1]):
#         for j in range(1):
            for re, i in enumerate(range(bases_shape[0])):
                if (re % 2) or (irrep_num==0):
                    bases_re.append(np.squeeze(conv._basisexpansion._modules[f"block_expansion_(\'irrep_0\', \'{irrep}\')"]._buffers['sampled_basis'][i,j,:,:]).reshape(kernel_size,kernel_size))
                else:
                    bases_im.append(np.squeeze(conv._basisexpansion._modules[f"block_expansion_(\'irrep_0\', \'{irrep}\')"]._buffers['sampled_basis'][i,j,:,:]).reshape(kernel_size,kernel_size))
                    
                if plot_bases:
                    plt.imshow(np.squeeze(conv._basisexpansion._modules[f"block_expansion_(\'irrep_0\', \'{irrep}\')"]._buffers['sampled_basis'][i,j,:,:]).reshape(kernel_size,kernel_size), cmap='gray')
                    plt.show()

    bases_re = torch.stack(bases_re, dim=0)
    bases_im = torch.stack(bases_im, dim=0)
    return bases_re, bases_im


## Complex dense layer using SO(2) bases

## For now I calculate the bases in the layer init, so that they are only calculated once, and these are non-trainable parameters of shape (num_bases, n_specled, n_specled)
## Set up weight matrices with shape (n_img, n_img, num_bases)
## At the moment I am multiplying the weight matrix with the bases everytime in the forward pass. I am not sure if there is a way to do this once in the init as it would significantly speed things up.

# Complex Dense Layer
class ComplexDenseSO2(torch.nn.Module):

#     def __init__(self, in_features, out_features, max_freq=24, num_rings=200, sigma=1.0,
    def __init__(self, in_features=180, out_features=180, max_freq=24, num_rings=200, sigma=1.0,
                 bias=True, **kwargs):
        super(ComplexDenseSO2, self).__init__(**kwargs)
        bases_re, bases_im = rad_bases(max_freq=max_freq, kernel_size=in_features, num_rings=num_rings, sigma=sigma, plot_bases=False)
        bases_re, bases_im = bases_re.view(-1,in_features*in_features), bases_im.view(-1,out_features*out_features)
        bases_im2 = []
        for i in range(len(bases_re)-len(bases_im)):
            bases_im2.append(torch.zeros(in_features*in_features))
        bases_im2 = torch.stack(bases_im2, dim=0)
        bases_im = torch.cat((bases_im2, bases_im))
        
        bases_re = bases_re / torch.max(bases_re) * 0.004054009220439023
        bases_im = bases_im / torch.max(bases_im) * 1.0211457198261614e-06
        
        bases_circ = channels_to_complex(torch.stack((bases_re,bases_im), dim=-1))
        
        bases = []
        for mode in bases_circ:
            mode = np.reshape(mode.numpy(), (in_features,in_features))
            mode = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(mode)))
            bases.append(torch.from_numpy(np.reshape(mode, (-1))))
            
        bases = torch.stack(bases, dim=0).type(torch.complex64)        
        self.bases = torch.nn.Parameter(bases, requires_grad=False)

        weight_re = torch.Tensor(len(bases))
        weight_im = torch.Tensor(len(bases))
        weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

        torch.nn.init.uniform_(self.weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))
        
    def forward(self, X):
        
        if X.dtype == torch.complex64:
            complex_X = X
        else:
            complex_X = channels_to_complex(X)
            complex_X = X = torch.flatten(complex_X, start_dim=-2)
            
        X = complex_X

        X_out = X @ torch.conj(torch.transpose(self.bases,0,1))
        
        X_out = X_out @ torch.diag_embed(self.weight)
        
        X_out = X_out @ self.bases
        X_out = complex_to_channels(X_out)
        X_re_out = X_out[...,0]
        X_im_out = X_out[...,1]
        
        output = torch.stack((X_re_out,X_im_out), dim=-1)
        output = channels_to_complex(output)
        
        return output
    
    
class NetSO2(torch.nn.Module):
    def __init__(self, img_size=180, max_freq=24, num_rings=200, sigma=1.0):
        super().__init__()
        self.cfc1 = ComplexDenseSO2(img_size, img_size, max_freq=max_freq, num_rings=num_rings, sigma=sigma)
#         self.hadamard2 = Hadamard2(180*180)
        
    def forward(self, x):
        x = self.cfc1(x)
#         x = self.hadamard2(x)
#         x = x.view(x.shape[0], -1, 28, 28)
        return x


class ComplexDenseSO2MixedDim(torch.nn.Module):

    def __init__(self, TM_name='TheoryTM', TM_name_in=None, Real_in=False, in_features=180, out_features=180, max_freq=24, num_rings=200, sigma=1.0, bias=True, **kwargs):
        super(ComplexDenseSO2MixedDim, self).__init__(**kwargs)
        self.Real_in = Real_in
        if TM_name_in == None:
            self.multires = False
        else:
            self.multires = True
        
        with open(f'data/{TM_name}.dat','rb') as f:
            TM_parts = pickle.load(f)
            
        if TM_name_in is not None:
            with open(f'data/{TM_name_in}.dat','rb') as f:
                TM_parts_in = pickle.load(f)
                
        bases_re_out, bases_im_out = rad_bases(max_freq=max_freq, kernel_size=out_features, num_rings=num_rings, sigma=sigma, plot_bases=False)
        bases_re_out, bases_im_out = bases_re_out.view(-1,out_features*out_features), bases_im_out.view(-1,out_features*out_features)
        bases_im2_out = []
        for i in range(len(bases_re_out)-len(bases_im_out)):
            bases_im2_out.append(torch.zeros(out_features*out_features))
        bases_im2_out = torch.stack(bases_im2_out, dim=0)
        bases_im_out = torch.cat((bases_im2_out, bases_im_out))
        
        bases_re_out = bases_re_out / torch.max(bases_re_out) * 0.004054009220439023
        bases_im_out = bases_im_out / torch.max(bases_im_out) * 1.0211457198261614e-06
        
        bases_circ_out = channels_to_complex(torch.stack((bases_re_out,bases_im_out), dim=-1))
        
        bases_out = []
        for mode in bases_circ_out:
            mode = np.reshape(mode.numpy(), (out_features,out_features))
            mode = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(mode)))
            bases_out.append(torch.from_numpy(np.reshape(mode, (-1))))
            
        bases_out = torch.stack(bases_out, dim=0).type(torch.complex64)
        num_so2_bases = len(bases_out)
        
        print(f'Computed SO2 bases : {bases_out.shape}')
        
        bases = []
        for mode in TM_parts["M"]:
            if TM_name=='TheoryTM':
                basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
            elif TM_name=='TheoryTM_256':
                basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
            elif TM_name=='TheoryTM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            elif TM_name=='TheoryTM_MM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_MM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            bases.append(torch.from_numpy(basis).type(torch.complex64))
        bases = torch.stack(bases, dim=0)
            
        if TM_name=='TheoryTM':
            bases = bases.view(-1,32400)
        elif TM_name=='TheoryTM_256':
            bases = bases.view(-1,65536)
        elif TM_name=='TheoryTM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_28':
            bases = bases.view(-1,784)
        elif TM_name=='TheoryTM_MM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_MM_28':
            bases = bases.view(-1,784)

        bases_out = torch.cat([bases_out, bases])
        print(f'Computed TM bases also : {bases_out.shape}')
        
        self.bases_out = torch.nn.Parameter(bases_out, requires_grad=False)
        
        bases_re_in, bases_im_in = rad_bases(max_freq=max_freq, kernel_size=in_features, num_rings=num_rings, sigma=sigma, plot_bases=False)
        bases_re_in, bases_im_in = bases_re_in.view(-1,in_features*in_features), bases_im_in.view(-1,in_features*in_features)
        bases_im2_in = []
        for i in range(len(bases_re_in)-len(bases_im_in)):
            bases_im2_in.append(torch.zeros(in_features*in_features))
        bases_im2_in = torch.stack(bases_im2_in, dim=0)
        bases_im_in = torch.cat((bases_im2_in, bases_im_in))
        
        bases_re_in = bases_re_in / torch.max(bases_re_in) * 0.004054009220439023
        bases_im_in = bases_im_in / torch.max(bases_im_in) * 1.0211457198261614e-06
        
        bases_circ_in = channels_to_complex(torch.stack((bases_re_in,bases_im_in), dim=-1))
        
        bases_in = []
        for mode in bases_circ_in:
            mode = np.reshape(mode.numpy(), (in_features,in_features))
            mode = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(mode)))
            bases_in.append(torch.from_numpy(np.reshape(mode, (-1))))
            
        bases_in = torch.stack(bases_in, dim=0).type(torch.complex64)
        bases_in = bases_in[:num_so2_bases]
        
        print(f'Computed SO2 bases : {bases_in.shape}')
        
        if TM_name_in is not None:
            bases = []
            for mode in TM_parts_in["M"]:
                if TM_name_in=='TheoryTM':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
                elif TM_name_in=='TheoryTM_256':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
                elif TM_name_in=='TheoryTM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                elif TM_name_in=='TheoryTM_MM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_MM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                bases.append(torch.from_numpy(basis).type(torch.complex64))
            bases = torch.stack(bases, dim=0)
#             if self.Real_in:
#                 bases_in = torch.abs(bases_in)
            if TM_name_in=='TheoryTM':
                bases = bases.view(-1,32400)
            elif TM_name_in=='TheoryTM_256':
                bases = bases.view(-1,65536)
            elif TM_name_in=='TheoryTM_224':
                bases = bases.view(-1,50176)
            elif TM_name_in=='TheoryTM_28':
                bases = bases.view(-1,784)
            elif TM_name_in=='TheoryTM_MM_224':
                bases = bases.view(-1,50176)
            elif TM_name_in=='TheoryTM_MM_28':
                bases = bases.view(-1,784)
        
        bases_in = torch.cat([bases_in, bases])
        print(f'Computed TM bases also : {bases_in.shape}')
        
        self.bases_in = torch.nn.Parameter(bases_in, requires_grad=False)

        weight_re = torch.Tensor(len(bases_in))
        weight_im = torch.Tensor(len(bases_in))
        weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
            
        torch.nn.init.uniform_(self.weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))

    def forward(self, X):

        if X.dtype == torch.complex64:
            pass
#             complex_X = X
        else:
            if not self.Real_in:
                X = channels_to_complex(X)
                X = torch.flatten(X, start_dim=-2)
            else:
                Xim = torch.zeros_like(X)
                X = channels_to_complex(torch.stack((X,Xim), dim=-1))

        X_out = X @ torch.conj(torch.transpose(self.bases_in,0,1))
        
        X_out = X_out @ torch.diag_embed(self.weight)
        
        X_out = X_out @ self.bases_out
        X_out = complex_to_channels(X_out)
        X_re_out = X_out[...,0]
        X_im_out = X_out[...,1]

        output = torch.stack((X_re_out,X_im_out), dim=-1)
        output = channels_to_complex(output)

        return output

# Complex Dense Layer
class ComplexDenseTMBasis(torch.nn.Module):

    def __init__(self, TM_name='TheoryTM', TM_name_in=None, Real_in=False, tm_full_mat=False, bias=True, block_diag_mat=0, **kwargs):
        super(ComplexDenseTMBasis, self).__init__(**kwargs)
        self.Real_in = Real_in
        self.tm_full_mat = tm_full_mat
        self.block_diag_mat = block_diag_mat
        if TM_name_in == None:
            self.multires = False
        else:
            self.multires = True
        
        with open(f'data/{TM_name}.dat','rb') as f:
            TM_parts = pickle.load(f)
            
        if TM_name_in is not None:
            with open(f'data/{TM_name_in}.dat','rb') as f:
                TM_parts_in = pickle.load(f)
        
        bases = []
        for mode in TM_parts["M"]:
            if TM_name=='TheoryTM':
                basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
            elif TM_name=='TheoryTM_256':
                basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
            elif TM_name=='TheoryTM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            elif TM_name=='TheoryTM_512':
                basis = np.conjugate(np.transpose(np.reshape(mode, (512,512))))
            elif TM_name=='TheoryTM_1024':
                basis = np.conjugate(np.transpose(np.reshape(mode, (1024,1024))))
            elif TM_name=='TheoryTM_MM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_MM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            bases.append(torch.from_numpy(basis).type(torch.complex64))
        bases = torch.stack(bases, dim=0)
            
        if TM_name=='TheoryTM':
            bases = bases.view(-1,32400)
        elif TM_name=='TheoryTM_256':
            bases = bases.view(-1,65536)
        elif TM_name=='TheoryTM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_512':
            bases = bases.view(-1,262144)
        elif TM_name=='TheoryTM_1024':
            bases = bases.view(-1,1048576)
        elif TM_name=='TheoryTM_28':
            bases = bases.view(-1,784)
        elif TM_name=='TheoryTM_MM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_MM_28':
            bases = bases.view(-1,784)
        self.bases = torch.nn.Parameter(bases, requires_grad=False)
        
        if TM_name_in is not None:
            bases_in = []
            for mode in TM_parts_in["M"]:
                if TM_name_in=='TheoryTM':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
                elif TM_name_in=='TheoryTM_256':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
                elif TM_name_in=='TheoryTM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                elif TM_name_in=='TheoryTM_512':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (512,512))))
                elif TM_name_in=='TheoryTM_1024':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (1024,1024))))
                elif TM_name_in=='TheoryTM_MM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_MM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                bases_in.append(torch.from_numpy(basis).type(torch.complex64))
            bases_in = torch.stack(bases_in, dim=0)
            if TM_name_in=='TheoryTM':
                bases_in = bases_in.view(-1,32400)
            elif TM_name_in=='TheoryTM_256':
                bases_in = bases_in.view(-1,65536)
            elif TM_name_in=='TheoryTM_224':
                bases_in = bases_in.view(-1,50176)
            elif TM_name_in=='TheoryTM_28':
                bases_in = bases_in.view(-1,784)
            elif TM_name_in=='TheoryTM_512':
                bases_in = bases_in.view(-1,262144)
            elif TM_name_in=='TheoryTM_1024':
                bases_in = bases_in.view(-1,1048576)
            elif TM_name_in=='TheoryTM_MM_224':
                bases_in = bases_in.view(-1,50176)
            elif TM_name_in=='TheoryTM_MM_28':
                bases_in = bases_in.view(-1,784)
            self.bases_in = torch.nn.Parameter(bases_in, requires_grad=False)

        if self.block_diag_mat != 0:
            weights = []
            for i in range(self.block_diag_mat):
                if i == 0: ## Need two sets of weights for off-diagonals for each side of the diagonal
                    weight_re = torch.zeros(len(bases)-i)
                    weight_im = torch.zeros(len(bases)-i)
                    weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
                    weight = torch.nn.Parameter(weight, requires_grad=True)
                    torch.nn.init.uniform_(weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))
                    weights.append(weight)
                else:
                    weight_re = torch.zeros(len(bases)-i)
                    weight_im = torch.zeros(len(bases)-i)
                    weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
                    weight = torch.unsqueeze(weight, dim=0).repeat(2, 1)
                    weight = torch.nn.Parameter(weight, requires_grad=True)
                    torch.nn.init.uniform_(weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))
                    weights.append(weight)
            self.weight = torch.nn.ParameterList(weights)
        else:
            if self.tm_full_mat:
                weight_re = torch.Tensor(len(bases),len(bases))
                weight_im = torch.Tensor(len(bases),len(bases))
            else:
                weight_re = torch.zeros(len(bases))
                weight_im = torch.zeros(len(bases))
            weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
            self.weight = torch.nn.Parameter(weight, requires_grad=True)
            torch.nn.init.uniform_(self.weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))

    def phase_amplitude_elu(self, z):
        return torch.nn.functional.elu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

    def forward(self, X):
        
        if X.dtype == torch.complex64:
            pass
        else:
            if not self.Real_in:
                X = channels_to_complex(X)
                X = torch.flatten(X, start_dim=-2)
            else:
                Xim = torch.zeros_like(X)
                X = channels_to_complex(torch.stack((X,Xim), dim=-1))
            
        if self.multires == False:
            X_out = X @ torch.conj(torch.transpose(self.bases,0,1))
        else:
            X_out = X @ torch.transpose(self.bases_in,0,1)
        
        if isinstance(self.weight, torch.nn.ParameterList):
            for i, w in enumerate(self.weight):
                if i == 0:
                    weight = torch.diag_embed(w)
                else:
                    w0 = w[0]
                    weight = weight + torch.diag_embed(w0, offset=i)
                    w1 = w[1]
                    weight = weight + torch.diag_embed(w1, offset=-i)
            X_out = X_out @ weight
        else:
            if self.tm_full_mat:
                X_out = X_out @ self.weight
            else:
                X_out = X_out @ torch.diag_embed(self.weight)
                
        output = X_out @ self.bases

        return output

# Complex Dense Layer
class ForInvTMBasis(torch.nn.Module):

    def __init__(self, TM_name='TheoryTM', TM_name_in=None, Real_in=False, tm_full_mat=False, bias=True, block_diag_mat=0, **kwargs):
        super(ForInvTMBasis, self).__init__(**kwargs)
        self.Real_in = Real_in
        self.tm_full_mat = tm_full_mat
        self.block_diag_mat = block_diag_mat
        if TM_name_in == None:
            self.multires = False
        else:
            self.multires = True
        
        with open(f'data/{TM_name}.dat','rb') as f:
            TM_parts = pickle.load(f)
            
        if TM_name_in is not None:
            with open(f'data/{TM_name_in}.dat','rb') as f:
                TM_parts_in = pickle.load(f)
        
        bases = []
        for mode in TM_parts["M"]:
            if TM_name=='TheoryTM':
                basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
            elif TM_name=='TheoryTM_256':
                basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
            elif TM_name=='TheoryTM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            elif TM_name=='TheoryTM_512':
                basis = np.conjugate(np.transpose(np.reshape(mode, (512,512))))
            elif TM_name=='TheoryTM_1024':
                basis = np.conjugate(np.transpose(np.reshape(mode, (1024,1024))))
            elif TM_name=='TheoryTM_MM_224':
                basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
            elif TM_name=='TheoryTM_MM_28':
                basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
            bases.append(torch.from_numpy(basis).type(torch.complex64))
        bases = torch.stack(bases, dim=0)

        if TM_name=='TheoryTM':
            bases = bases.view(-1,32400)
        elif TM_name=='TheoryTM_256':
            bases = bases.view(-1,65536)
        elif TM_name=='TheoryTM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_512':
            bases = bases.view(-1,262144)
        elif TM_name=='TheoryTM_1024':
            bases = bases.view(-1,1048576)
        elif TM_name=='TheoryTM_28':
            bases = bases.view(-1,784)
        elif TM_name=='TheoryTM_MM_224':
            bases = bases.view(-1,50176)
        elif TM_name=='TheoryTM_MM_28':
            bases = bases.view(-1,784)
        self.bases = torch.nn.Parameter(bases, requires_grad=False)
        
        if TM_name_in is not None:
            bases_in = []
            for mode in TM_parts_in["M"]:
                if TM_name_in=='TheoryTM':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (180,180))))
                elif TM_name_in=='TheoryTM_256':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (256,256))))
                elif TM_name_in=='TheoryTM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                elif TM_name_in=='TheoryTM_512':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (512,512))))
                elif TM_name_in=='TheoryTM_1024':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (1024,1024))))
                elif TM_name_in=='TheoryTM_MM_224':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (224,224))))
                elif TM_name_in=='TheoryTM_MM_28':
                    basis = np.conjugate(np.transpose(np.reshape(mode, (28,28))))
                bases_in.append(torch.from_numpy(basis).type(torch.complex64))
            bases_in = torch.stack(bases_in, dim=0)
            if TM_name_in=='TheoryTM':
                bases_in = bases_in.view(-1,32400)
            elif TM_name_in=='TheoryTM_256':
                bases_in = bases_in.view(-1,65536)
            elif TM_name_in=='TheoryTM_224':
                bases_in = bases_in.view(-1,50176)
            elif TM_name_in=='TheoryTM_28':
                bases_in = bases_in.view(-1,784)
            elif TM_name_in=='TheoryTM_512':
                bases_in = bases_in.view(-1,262144)
            elif TM_name_in=='TheoryTM_1024':
                bases_in = bases_in.view(-1,1048576)
            elif TM_name_in=='TheoryTM_MM_224':
                bases_in = bases_in.view(-1,50176)
            elif TM_name_in=='TheoryTM_MM_28':
                bases_in = bases_in.view(-1,784)
            self.bases_in = torch.nn.Parameter(bases_in, requires_grad=False)

            weight_re = torch.zeros(len(bases))
            weight_im = torch.zeros(len(bases))
            weight = channels_to_complex(torch.stack((weight_re,weight_im), dim=-1))
            self.weight = torch.nn.Parameter(weight, requires_grad=True)
            torch.nn.init.uniform_(self.weight,-np.sqrt(1/len(bases)),np.sqrt(1/len(bases)))

    def forward(self, X):

        if X.dtype == torch.complex64:
            pass
        else:
            if not self.Real_in:
                X = channels_to_complex(X)
                X = torch.flatten(X, start_dim=-2)
            else:
                Xim = torch.zeros_like(X)
                X = channels_to_complex(torch.stack((X,Xim), dim=-1))

        X_out = X @ torch.conj(torch.transpose(self.bases,0,1))
        X_out = X_out @ torch.diag_embed(1./self.weight)
        output_speckled = X_out @ self.bases_in
        
        X_out = output_speckled @ torch.conj(torch.transpose(self.bases_in,0,1))
        X_out = X_out @ torch.diag_embed(self.weight)
        output = X_out @ self.bases

        return output_speckled, output
    
class NetTMBasis(torch.nn.Module):
    def __init__(self, TM_name='TheoryTM', TM_name_in=None, Real_in=False, tm_full_mat=False, block_diag_mat=0):
        super().__init__()
        self.Real_in = Real_in
        self.cfc1 = ComplexDenseTMBasis(TM_name, TM_name_in, Real_in, tm_full_mat, block_diag_mat=block_diag_mat)
#         self.cfc1 = ForInvTMBasis(TM_name, TM_name_in, Real_in, tm_full_mat, block_diag_mat=block_diag_mat)
#         self.cfc1 = ComplexDenseSO2MixedDim(TM_name, TM_name_in, Real_in, 224, 28, max_freq=24, num_rings=200, sigma=1.0)
#         if self.Real_in:
#             self.hadamard2 = Hadamard2(224*224)
        
    def forward(self, x):
        if x.dtype == torch.complex64:
            pass
        else:
            if not self.Real_in:
                x = channels_to_complex(x)
                x = torch.flatten(x, start_dim=-2)
            else:
                xim = torch.zeros_like(x)
                x = channels_to_complex(torch.stack((x,xim), dim=-1))
                
#         if self.Real_in:
#             x = self.hadamard2(x)
        xspeck, x = self.cfc1(x)
#         x = self.hadamard2(x)
        return xspeck, x

class NetTMBasis_FI(torch.nn.Module):
    def __init__(self, TM_name='TheoryTM', TM_name_in=None, Real_in=False, tm_full_mat=False, block_diag_mat=0):
        super().__init__()
        self.Real_in = Real_in
        self.cfc1 = ForInvTMBasis(TM_name, TM_name_in, Real_in, tm_full_mat, block_diag_mat=block_diag_mat)

#         if self.Real_in:
#             self.hadamard2 = Hadamard2(224*224)
        
    def forward(self, x):
        if x.dtype == torch.complex64:
            pass
        else:
            if not self.Real_in:
                x = channels_to_complex(x)
                x = torch.flatten(x, start_dim=-2)
            else:
                xim = torch.zeros_like(x)
                x = channels_to_complex(torch.stack((x,xim), dim=-1))
                
#         if self.Real_in:
#             x = self.hadamard2(x)
        xspeck, x = self.cfc1(x)
#         x = self.hadamard2(x)
        return xspeck, x
