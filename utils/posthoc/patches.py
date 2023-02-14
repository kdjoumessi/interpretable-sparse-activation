import torch
import numpy as np

#------------------------
def get_patches(img, patchsize=33, stride=8):
    '''
        img (numpy): initial input image of shape (bs, H, W, C)
    '''
    image = img.astype(np.float32)
    input  = torch.from_numpy(image).cuda()
    patches = input.unfold(1, patchsize, stride).unfold(2, patchsize, stride)
    patches = patches.contiguous().view((-1, 3, patchsize, patchsize))
    
    return patches.cpu()


#------------------------
def topK_patches(scores, size=60, dx=-2, dy=2, p=4, k=5, max_=16, threshold=None):
    '''
        description: use a trick to extract a unique local patch with the highest score within a cluster of 16 posible patches
        
        parameters: 
            - scores (list): flatten logits 3600 elements
            - size (int): the size of the logit. it will be used to map a patch idx from the low to the high resolution
            - k (int): the number of non overlapping patches to select
            - threshold (int): the threshold above which we should select patch
            - (dx, dx): how to center the current patch or how to create the local area
            - p (int): the value is 4 since one patch can overlay in 3 other in the same direction
    '''
    values, idx = torch.topk(scores, len(scores))
    values = values.int().tolist()
    idx    = idx.int().tolist()
    #print('values len: ', len(values))
    
    dic_index = {}
    dic_val = {}
    top_patch_idx = []  # list of non-overlapping indices
    top_patch_val = []  # corresponding list of non-overlapping scores
    
    i = 0
    overlapping_idx = [] # list of overlaping indices ie indices in the neighborhood of a selected patch
    while ((i < len(scores) and len(top_patch_idx) < k) and (values[i] > threshold)):
        #print(f' values={values}, i={i}, values[i]={values[i]}')
        val = values[i]; index = idx[i]         # we select a local top patch with its corresponding index        
        
        # we add the current index to the final list if it is not in the neighborhood of a selected index
        if index not in overlapping_idx:  
            top_patch_idx.append(index)
            top_patch_val.append(val)
        
        # how to manage extrem cases where after adding dc or dy the final index is out of bounds ?
        # 1+8 => 9, 9+8 => 17, 17+8 => 25, 25+8 => 33: for one patch we can have 16 other overlapping on it
        org_row = index // size;           org_col = index % size
        init_row_idx = org_row + dx;        init_col_idx = org_col + dy         
        
        coord = (init_row_idx, init_col_idx)
        #print('original coordinate: ', (org_row, org_col))
        #print('init local coordinate: ', coord)
        
        # for each x, fix x and decrease y index : this computes all the posible neighborhood of a given patch
        coords = [(coord[0]+i, coord[1]-j) for i in range(p) for j in range(p)] # store the vals in a dict
        #print('coords: \n', coords)
        
        # we construct the corresponding index from the low to the high dim space
        tmp_1D_idx_from_coord = [(elt[0] * size) + elt[1] for elt in coords] 
        dic_index[index] = tmp_1D_idx_from_coord  # list of neighborhood of a given patch
        dic_val[index] = val
        
        #print('neighboring coordinate: ', tmp_1D_idx_from_coord)
        #print('index: ', index)
        #print('list: ', tmp_1D_idx_from_coord)
        #tmp_1D_idx_from_coord.remove(index)
        overlapping_idx += tmp_1D_idx_from_coord
        i += 1
        
        dic_index['overlap'] = overlapping_idx
        
        if len(top_patch_idx) > max_:
            break
        #break
    
    return top_patch_idx, top_patch_val, dic_index, dic_val