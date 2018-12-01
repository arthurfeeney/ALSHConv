
import torch

def print_mean_feature_map(conv_output):
    r'''
    this looks at the mean of the output of each filter in a convolution. 
    Typically, large values are good.
    '''
    mean_list = torch.mean(conv_output, dim=0)
    print('all means:')
    print(mean_list.max())
    print(mean_list.min())
    
    nz_mean_list = conv_output[conv_output > 0].mean(dim=0) 
    print('nonzero means:')
    print(nz_mean_list.max())
    print(nz_mean_list.min())
    

