#-------------------------------------------
# Source code for backdrop pytorch implementation
#
# By:  Siavash Golkar, Kyle Cranmer
#-------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# ______________________________________________
# Defining the layers and network
# ----------------------------------------------

from torch.autograd import Function

# This is the place masking happens.
class mask_func(Function):  # Define a torch.autograd function
    
    # Static method is crucial for passing the appropriate contexts.
    #Forward is called through (instance.apply).
    @staticmethod
    def forward(ctx, x, mask_forward, mask_backward):
        
        # Saving the backward mask for the backprop pass.
        ctx.save_for_backward(mask_backward)
        
        return x.new(x*mask_forward)

    @staticmethod
    def backward(ctx, grad_output):
        
        # Recalling the saved backward mask tensor.
        mask_backward, = ctx.saved_tensors
        grad_output = grad_output.data

        return Variable(grad_output * mask_backward), None, None


#Defining a placeholder module so we can use it inside nn.sequential.
class mask_mod(nn.Module): 
    """
    mask_dims is the list of dimensions along which we would
    want to implement masking.
    """
    def __init__(self, mask_prob, mask_dims, mask_el = None):
        super(mask_mod, self).__init__()
        # Define an instance of mask_func
        self.masker = mask_func()
        self.mask_prob = mask_prob
        self.verbose = False
        self.mask_el = mask_el
        self.mask_dims = mask_dims
        
    def forward(self, x):   
        
        s = x.shape # The shape for the masking.
        mask_shape = [s[i] if i in self.mask_dims else 1 for i in range(max(self.mask_dims)+1)]
        mask_of_ones = torch.ones(mask_shape).type_as(x.data) 
        
        if self.mask_el != None:
            mask_backward = torch.zeros_like(mask_of_ones).type_as(x.data) 
            loc = []; count=0
            for i in range(len(mask_shape)):
                if mask_shape[i]>0: 
                    loc.append(self.mask_el[count])
                    count+=1
                else: loc.append(0)
            mask_backward[tuple(loc)] = 1
        else:
            
            # torch.bernouli(X) gives a bernouli distribution for each component of X
            # with p_ij = X_ij. i.e. X_ij = 1, always gives 1. => p = 1 - mask_prob
            mask_backward = torch.bernoulli(mask_of_ones.type_as(x.data) * (1 - self.mask_prob))

            if mask_backward.abs().max() ==0: #Making sure the mask does not return all zeros.
                loc = [np.random.randint(s[i]) if i in self.mask_dims else 0 for i in range(max(self.mask_dims)+1)]
                mask_backward[tuple(loc)] = 1
            
        
        # We need to normalize to make sure the backward pass gives a similar gradient.
        # Norm based on actual mask
        mask_backward = (Variable(mask_backward)/(mask_backward).sum() * np.prod(mask_shape).astype(float)).type_as(x.data) 
        
        if self.verbose:
            print('Generated mask with shape: ', mask_backward.shape)
            print('The generate mask is: ', mask_backward.squeeze().data.cpu().numpy())
            print('-'*50, '\n')
        
        # Calling the defined mask_func instance.
        return self.masker.apply(x, mask_of_ones, mask_backward)