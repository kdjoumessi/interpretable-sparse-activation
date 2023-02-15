import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils.posthoc.patches import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

#------------------------
def plot_EyePACS(img, label, size=(6, 6)):
    '''
    plots the images
    
        Parameters:
            - img (np.array): (bs,H,W,C)
            - label (int): class level
    '''
    with plt.style.context('./utils/posthoc/plot_style.txt'):    
        fig = plt.subplots(figsize=size)
        n = len(img)
        
        for i in range(n):
            plt.subplot(1, n+1, i+1) # (row nber, col nber, image nber)
            plt.imshow(img[i])
            plt.title(f'level = {label[i]}')
            plt.axis('off')
        plt.plot()


#------------------------
def plot_probabilities(imgs, df, proba, description, binary=False, on_same_plot=False, size=(6,6)):
    ''' 
    Plot image with corresponding probabilities to visualize an image and its predicted classes

        inputs: 
            - imgs (np.array): sample image -> (bs, H,W,C)
            - df: dataframe containing images
            - proba (Torch.Tensor): list of tensors prediction from each model [res, baseline_bag, dense_bag, sparse_bag]
            - binary (bool): if false => multiclass task
            - on_same_plot (bool): plot the proba on the same plot or not
    '''
    dic = {}
    ncol = len(proba) # nber of model
    bs = len(proba[0])
    y_label_bin = ['No_rDR', 'rDR']
    y_label_mult = ['0-NoDR', '1-MILD', '2-Moderate', '3-Severe', '4-Proliferative']
    class_num, ylabel = (2, y_label_bin) if binary else (5, y_label_mult) 
    
    for i in range(ncol): # for each model   
        preds = proba[i]  # output of one model
        tmp_list = []
        for j in range(bs): # for each images
            pred = preds[j]
            top_score, class_label = torch.topk(pred, 1) # top1 prediction score and class index
            top_score_val, class_label_val = top_score.item(), class_label.item()        
            tmp_list.append([pred, class_label_val, top_score_val])
        dic[i+1] = tmp_list # 1=ResNet, 2=baseline_bagnet, 3=dense_bagnet, 4=sparse_bagnet 

    #fs = 8
    with plt.style.context('./utils/posthoc/plot_style.txt'):    
        fig = plt.figure(figsize=size, layout='constrained')
        
        k, j = 0, 0
        ncol = 2 if on_same_plot else (ncol + 1)  
        for i in range(bs): # for each sample
            #'''
            j += 1
            ax = fig.add_subplot(bs, ncol, j) 
            ax.imshow(np.clip(imgs[i], 0, 1))
            ax.axis('off')
            if binary:
                ax.set_title(f'Image = {df.image.iloc[i]}, Label= {df.onset2.iloc[i]}')
            else:
                ax.set_title(f'Image = {df.image.iloc[i]}, Label= {df.level.iloc[i]}')
            k=1
            #'''
            if on_same_plot:
                j += 1
                ax = fig.add_subplot(bs, ncol, j)
                width = 0.2
                ind = np.arange(class_num)
                ax.barh(ind, dic[1][i][0], width, color='r', label='ResNet')
                ax.barh(ind+width, dic[2][i][0], width, color='b', label='dense BagNet')
                ax.barh(ind+width*2, dic[3][i][0], width, color='g', label='sparse BagNet')
                
                ax.set_aspect(0.1)
                ax.set_yticks(np.arange(class_num))
                ax.set_yticklabels(np.arange(0,class_num)) # fontsize=fs
                ax.set_xlim(0, 1.0)
                plt.xticks() # fontsize=fs
                if i==0:
                    ax.legend()  # fontsize=fs
            else:
                for key, value in dic.items(): # for each model where the key is the model id and the values the list of predicted output
                    j += 1
                    ax = fig.add_subplot(bs, ncol, j)
                    ax.barh(np.arange(class_num), value[i][0]) # plot the proba output for each class
                    ax.set_aspect(0.1)      
                    ax.set_yticks(np.arange(class_num))
                    if (k == 1) and binary:                        
                        ax.set_yticklabels(ylabel)
                        k +=1
                    else:
                        ax.set_yticklabels(ylabel)
                        
                    ax.set_title(f'{description[key-1]}: pred= {value[i][1]}, prob= {round(value[i][2], 3)}')
                    ax.set_xlim(0, 1.0)
    return dic

#------------------------
def plot_binary_GradCAM_heatmap(org_img, list_dics, df, y_label, proba=None, plot_prob=False, size=(21, 21), bag_alpha=.25, cam_alpha=.6):
    '''
        description: plot the image, gradCAM and heatmap(s)
        - org_imgs:
        - list_dics:
        - df
        - N is the number of heatmap to be plotted
    '''
    with plt.style.context('./utils/posthoc/plot_style.txt'): 
        fig = plt.figure(figsize=size, layout='constrained') #
        
        bs = len(org_img)        
        j = 0
        ncol = len(list_dics) + 1
        if proba:
            resnet, dense, sparse = proba[1], proba[2], proba[3]

        for i in range(bs):
            dense_heat, sparse_heat, gradcam = list_dics[0][i][1], list_dics[1][i][1], list_dics[2][i][0]
            vmax = dense_heat['vmax'] 
            extent, cmap, cmap_2, overlay = sparse_heat['extent'], sparse_heat['cmap'], sparse_heat['cmap_original'], sparse_heat['overlay']

            j += 1
            ax = fig.add_subplot(bs, ncol, j)
            ax.imshow(np.clip(org_img[i], 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.set_ylabel(y_label[i])

            j += 1
            ax = fig.add_subplot(bs, ncol, j)
            ax.imshow(gradcam, extent=extent, interpolation='none', alpha=cam_alpha, cmap=cmap)
            ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_2, alpha=bag_alpha)   
            if (j==2):
                plt.title('ResNet')
            if plot_prob:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xlabel('$p={:0.3f}$'.format(resnet[i][0][1]))
            else:
                ax.axis('off')

            j += 1
            ax = fig.add_subplot(bs, ncol, j)
            img = ax.imshow(dense_heat['heatmap'], extent=dense_heat['extent'], interpolation='none', cmap=dense_heat['cmap'], vmin=-vmax, vmax=vmax)
            ax.imshow(dense_heat['overlay'], extent=dense_heat['extent'], interpolation='none', cmap=dense_heat['cmap_original'], alpha=bag_alpha)
            if j==3:
                plt.title('dense BagNet')
            if plot_prob:
                ax.set_xticks([])
                ax.set_yticks([])      
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xlabel('$p={:0.3f}$'.format(dense[i][0][1]))  
            else:
                ax.axis('off')        

            j += 1
            ax = fig.add_subplot(bs, ncol, j)
            img = ax.imshow(sparse_heat['heatmap'], extent=sparse_heat['extent'], interpolation='none', cmap=sparse_heat['cmap'], vmin=-vmax, vmax=vmax)
            ax.imshow(sparse_heat['overlay'], extent=sparse_heat['extent'], interpolation='none', cmap=sparse_heat['cmap_original'], alpha=bag_alpha)
            if (j==4):
                plt.title('sparse BagNet')
            if plot_prob:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xlabel('$p={:0.3f}$'.format(sparse[i][0][1]))
            else:
                ax.axis('off')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.03)
            plt.colorbar(img, cax=cax, use_gridspec=False)
        #plt.tight_layout()


#------------------------
def plot_localization(cfg, img, fmap, data_path, annotated_img=None, fig_size=(14,14), save=False, k=16, print_score=True):
    '''
        - cfg
        - sparse_fmap
        - data_dir
        - annotation_dir
        - print_score
    '''
    
    image = np.expand_dims(img, axis=0)
    patches = get_patches(image).permute(0,2,3,1) 
    flatten_fmap = torch.from_numpy(fmap.flatten())
    
    n = len(flatten_fmap)
    #scores, indices = torch.topk(flatten_fmap, k)
    #scores, indices = scores.tolist(), indices.tolist()
    
    indices, scores, dic_idx, dic_val = topK_patches(flatten_fmap, k=n, max_=16, threshold=0)
    plot_image_with_patches(img, patches, indices, scores, fig_size, ps=33, s=8, size=60, print_score=print_score)

#------------------------
def plot_image_with_patches(img, patches, indices, scores, fig_size, ps=33, s=8, size=60, print_score=True):
    '''
        plot the image with the candidate patches based on the top logit score
        
        parameters:
            - img (np.array): the image to be ploted and from which the patches are extracted
            - indices (list): the indices of the extracted top paches (in [0-3600])
            - scores (list): the scores of the extracted top patches
            - ps (int): the patch size (by default is 33)
            - size (int): the size of the logit. it will be used to map a patch idx from the low to the high resolution
    '''
    fig = plt.figure(figsize=fig_size) 
    gs = fig.add_gridspec(14, 14)
    ax1 = fig.add_subplot(gs[0:8, :8])
    ax1.imshow(img) 
    if print_score:
        pass
        #ax1.set_title(f'image: {image}', fontsize=15)
    ax1.set_aspect('equal', adjustable='box')
    ax1.axis('off')
        
    
    idx = []        
    for i in indices: # from low to high resolution index
        row_idx = (i // size)     # get the row coordinate for the flatten index on the (size, size) cordinate 
        start_row = row_idx * s   # get the corresponding index in the high dimension space: (512, 512)
        end_row = start_row + ps 
        
        col_idx = i % size        # get the col coordinate for the flatten index on the (size, size) cordinate 
        start_col = col_idx * s   # get the corresponding index in the high dimension space: (512, 512)    
        end_col = start_col + ps   
        
        idx.append((start_row, start_col))
        rect = matplotlib.patches.Rectangle((start_col, start_row), ps, ps, linewidth=1.3, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)      
    
    ax2 = fig.add_subplot(gs[0:2, 8:10])
    ax3 = fig.add_subplot(gs[0:2, 10:12])
    ax4 = fig.add_subplot(gs[0:2, 12:14])
        
    ax5 = fig.add_subplot(gs[2:4, 8:10])
    ax6 = fig.add_subplot(gs[2:4, 10:12])
    ax7 = fig.add_subplot(gs[2:4, 12:14])
        
    ax8 = fig.add_subplot(gs[4:6, 8:10])
    ax9 = fig.add_subplot(gs[4:6, 10:12])
    ax10 = fig.add_subplot(gs[4:6,12:14])
        
    ax11 = fig.add_subplot(gs[6:8, 8:10])    
    ax12 = fig.add_subplot(gs[6:8, 10:12])
    ax13 = fig.add_subplot(gs[6:8, 12:14])
    axes_list = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13]
    
    k = 0
    x=1
    y=11
    k_num = False
    
    for ax in axes_list:
        ax.axis('off')
    
    for ax, i, score_ in zip(axes_list, indices, scores):
        ax.set_aspect('equal', adjustable='box')
        #ax1.text(idx[k][1]+x, idx[k][0]+y, f'{k+1}', fontsize=8) # plot number on the image patch
        ax.imshow(np.clip(patches[i,:,:,:].numpy(), 0, 1))
        if print_score:
            if k_num:
                print('yes')
                ax.set_title(f'k={k+1}, score={score_}', size='small')
                k += 1
            else:
                print('false')
                ax.set_title(f'n={i}, score={score_}', size='small')
        plt.plot()

#------------------------
def plot_multi_class_GradCAM_heatmap(org_img, list_dics, df, label, size=(21, 21), color_map=False, bag_alpha=.25, cam_alpha=.6):
    '''
        description: plot the image, gradCAM and heatmap(s)
        - org_imgs:
        - list_dics:
        - df
        - N is the number of heatmap to be plotted
    '''
    with plt.style.context('./utils/posthoc/plot_style.txt'): 
        fig = plt.figure(figsize=size, layout='constrained') # , 

        bs = len(org_img)
        j = 0
        ncol = 7
        nber_row = bs*2
        for i in range(bs):
            dense_heat, sparse_heat, gradcam = list_dics[0][i], list_dics[1][i], list_dics[2][i][0]
            vmax = dense_heat[df.iloc[i]['level']]['vmax'] 

            extent, cmap, cmap_2, overlay = sparse_heat[i]['extent'], sparse_heat[i]['cmap'], sparse_heat[i]['cmap_original'], sparse_heat[i]['overlay']
            j += 1
            ax = fig.add_subplot(nber_row, ncol, j)
            ax.imshow(np.clip(org_img[i], 0, 1))
            plt.title(label[i])
            ax.axis('off')

            j += 1
            ax = fig.add_subplot(nber_row, ncol, j)
            ax.imshow(gradcam, extent=extent, interpolation='none', alpha=cam_alpha, cmap=cmap)
            ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_2, alpha=bag_alpha)
            if j==2:
                plt.title('GradCAM')
            ax.axis('off')

            for k in range(5):
                j += 1
                ax = fig.add_subplot(nber_row, ncol, j)
                img = ax.imshow(dense_heat[k]['heatmap'], extent=dense_heat[k]['extent'], interpolation='none', cmap=dense_heat[k]['cmap'], vmin=-vmax, vmax=vmax)
                ax.imshow(dense_heat[k]['overlay'], extent=dense_heat[k]['extent'], interpolation='none', cmap=dense_heat[k]['cmap_original'], alpha=bag_alpha)
                if j==3:
                    plt.title('healthy')
                elif j==4:
                    plt.title('mild')
                elif j==5:
                    plt.title('moderate')
                elif j==6:
                    plt.title('severe')
                elif j==7:
                    plt.title('proliferate')
                ax.axis('off')

                if (color_map) and (k==4):
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(img, cax=cax)


            for k in range(2):
                j += 1
                ax = fig.add_subplot(nber_row, ncol, j)
                ax.imshow(np.ones((255, 255, 3)))
                ax.axis('off')

            for k in range(5):
                j += 1
                ax = fig.add_subplot(nber_row, ncol, j)
                img = ax.imshow(sparse_heat[k]['heatmap'], extent=sparse_heat[k]['extent'], interpolation='none', cmap=sparse_heat[k]['cmap'], vmin=-vmax, vmax=vmax)
                ax.imshow(sparse_heat[k]['overlay'], extent=sparse_heat[k]['extent'], interpolation='none', cmap=sparse_heat[k]['cmap_original'], alpha=bag_alpha)
                if (i==0) and (k==0):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.set_ylabel('sparse BagNet')                
                else:
                    ax.axis('off')

                if (color_map) and (k==4):
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(img, cax=cax)
                    #plt.subplots_adjust(wspace=0.03, hspace=- 0.5)