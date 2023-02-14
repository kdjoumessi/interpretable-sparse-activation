import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from matplotlib import pyplot as plt
from skimage import feature, transform
from pytorch_grad_cam.utils.image import show_cam_on_image

dense_features_maps = {}
def get_dense_features_maps(name):
    def hook(model, input, output):
        dense_features_maps[name] = output.detach().cpu().numpy()
    return hook

sparse_features_maps = {}
def get_sparse_features_maps(name):
    def hook(model, input, output):
        sparse_features_maps[name] = output.detach().cpu().numpy()
    return hook

####### Heatmap + saliency map ################
def get_feature_maps_from_inference(model, img, binary=True):
    '''
        extract the low resolution heatmap from inference

        parameters:
            - model (model / list of models)
            - ts_img (torch.tensor)
            - all_maps (bool): 

        outputs:
            - dic: dictionary containing low resolution heatmap for each model
    '''
    dic = {}
    preds = {}
    layers_name = 'avgpool'  # feature map layer from which the feature map is extracted before the GAP leading to the final classification
    dense_bag, sparse_bag = model['dense'], model['sparse']
    dense_bag.conv2.register_forward_hook(get_dense_features_maps(layers_name))
    sparse_bag.conv2.register_forward_hook(get_sparse_features_maps(layers_name))   
    
    preds['dense'] = dense_bag(img)
    preds['sparse'] = sparse_bag(img)
    
    if binary:
        dic['dense'] = dense_features_maps[layers_name]#[0][1]
        dic['sparse'] = sparse_features_maps[layers_name]
    else:
        raise NotImplementedError("Not yet implemented")
        #dic[i] = activation[name][0]
        
    del dense_bag; 
    del sparse_bag
    return dic, preds


#------------------------
def get_bagnet_heatmap_and_GradCAM(model, logits, imgs, mode=None, cmap='RdBu_r', dilation=0.5, percentile=99, alpha=.25):
    '''
        descprition: compute the GradCAM and the high resolution heatmap
        
        parameters:
            data_path:
            image_file:
            model: any resnet base model with FCL as the classification layer
            imgs (list) : [img, ts_img]
                - ts_img (torch.tensor): tensor image of shape (bs, C, H, W): the one use for inference
                - img (np.array): initial image to extract overlay mask of shape (bs, H,W,C) 
            logits (np.array): feature maps from the forward pass
    ''' 
    ts_img = imgs[1]
    image = imgs[0]#[0]
    
    image = image.transpose([0,3,1,2])
    
    # upsampling the heatmap from 60x60 to 512x512
    if mode == 'bilinear':
        m = torch.nn.Upsample(size=[512, 512], mode=mode, align_corners=True)
    elif mode == 'nearest':
        m = torch.nn.Upsample(size=[512, 512], mode=mode)
    else:
        NotImplemented('incorrect mode') 
    
    dense_feature_map, dic_dense_logits = {}, {}
    sparse_feature_map, dic_sparse_logits = {}, {}
    
    dx, dy = 0.05, 0.05
    def get_class_wise_feature_maps(fmaps, img):
        # from low to high resolution feature maps
        np_logit = fmaps[np.newaxis, np.newaxis, :, :] # adding two new dimension to the low resolution heatmap
        Upsample = m(torch.from_numpy(np_logit))
        hd_logit = Upsample[0,0].numpy()
        
        # BagNet heatmap
        xx = np.arange(0.0, hd_logit.shape[1], dx)
        yy = np.arange(0.0, hd_logit.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_original = plt.get_cmap('Greys_r').copy()
        cmap_original.set_bad(alpha=0)
        overlay = None
        
        original_greyscale = np.mean(img.transpose([1,2,0]), axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', channel_axis=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
        
        abs_max = np.percentile(np.abs(hd_logit), percentile)
        abs_min = abs_max
        class_dic = {'heatmap': hd_logit, 'overlay': overlay, 'extent': extent, 'cmap': cmap,
                'cmap_original': cmap_original, 'vmin': abs_min, 'vmax': abs_max}
        return class_dic, hd_logit    
        
    for key in logits:
        for i in range(len(logits[key])): # for each image
            fmap = logits[key][i] 
            image_fmap = {}
            image_hd_logit = {}
            for j in range(len(fmap)): # for each class or feature map
                image_fmap[j], image_hd_logit[j] = get_class_wise_feature_maps(fmap[j], image[i])
                
            if key == 'dense':
                dense_feature_map[i] = image_fmap
                dic_dense_logits[i] = image_hd_logit
            else:
                sparse_feature_map[i] = image_fmap
                dic_sparse_logits[i] = image_hd_logit
        
    # GradCAM
    dic_resnet = {}
    # org_img: numpy array of size (H, W, C): use to overlay the CAM on it
    target_layers = [model.layer4[-1]] # extract the last conv block in the penultimate layer
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    
    for i in range(len(ts_img)):
        t_img = torch.unsqueeze(ts_img[i], dim=0) 
        grayscale_cam = cam(input_tensor=t_img) 
        grayscale_cam = grayscale_cam[0, :] #(C, H, W)  => only the GradCAM whithout overlaying on the image
        visualization = show_cam_on_image(imgs[0][i], grayscale_cam, use_rgb=True)
        dic_resnet[i] = (grayscale_cam, visualization)
        
    return (dense_feature_map, dic_dense_logits), (sparse_feature_map, dic_sparse_logits), dic_resnet
