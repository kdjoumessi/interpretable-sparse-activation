import os
import torch
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from datetime import datetime

from modules.loss import *
from modules.scheduler import *
from utils.func import save_weights, print_msg, inverse_normalize, select_target_type, matplotlib_roccurve, matplotlib_prec_recall_curve, plot_conf_matrix

# hooks function to get the final activation for sparsity regularization
points = [(0.05, 0.80), (0.20, 0.85)] # (TNR, TPR) humain performaces on DR grading
activation = {}
def get_activation(name, version):
    def hook(model, input, output):
        if version == 'v2':
            activation[name] = output
        else:
            activation[name] = input
    return hook

def train(cfg, model, train_dataset, val_dataset, estimator, logger=None):
    
    device = cfg.base.device
    optimizer = initialize_optimizer(cfg, model)
    weighted_sampler = initialize_sampler(cfg, train_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler)

    print('train loader size: ', len(train_loader))
    print('val loader size: ', len(val_loader))

    save_path = cfg.save_paths.model 
    
    # start training
    model.train()

    if 'bagnet' in cfg.train.network:
        print('BagNet version: ', cfg.train.version)
        print('Training BagNet {} with spasity constraint: {}'.format(cfg.train.version, cfg.train.sparsity))
        if cfg.train.sparsity: # this define how to get the activation for regularization/sparsity
            if cfg.train.version == 'v2': # get the output of a given layer
                model.conv2.register_forward_hook(get_activation('avgpool', version='v2'))
            else: # how to get the input of the FC layer to regularize on it
                model.fc.register_forward_hook(get_activation('avgpool', version='v1'))

    max_indicator, max_bin_indicator, max_auc, max_sens, max_spe, max_pre = -1, -1, -1, -1, -1, -1
    min_loss_indicator = 10
    kappa_indicator = -100
    avg_loss, avg_val_loss, avg_acc, avg_kappa, avg_bin_acc = 0, 0, 0, 0, 0
    train_loss, train_acc, train_kappa, lr, train_bin_acc   = [], [], [], [],[]
    val_loss, val_acc, val_kappa, val_bin_acc = [], [], [], []
    train_auc, train_auprc, train_sens, train_prec, train_spec = [], [], [], [], []
    val_auc, val_auprc, val_sens, val_prec, val_spec = [], [], [], [], []

    # batch reg loss params to monitor the regularization contribution
    CE_loss, l1_reg, reg = [], [], []
    # epoch reg saving
    e_CE_loss, e_reg, e_l1_reg = [], [], []

    for epoch in range(1, cfg.train.epochs + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss, epoch_CE_loss, epoch_act_map, epoch_l1_reg = 0, 0, 0, 0
        avg_epoch_l1_reg, avg_act_map, avg_CE_loss = 0, 0, 0

        estimator.reset()
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            # forward
            y_pred = model(X)

            if cfg.train.sparsity:
                linear_fts = activation['avgpool']
                sparsity_constraint = torch.norm(linear_fts[0], 1) # l1_norm

                # batch loss
                if cfg.train.batch_reg_record:
                    l = loss_function(y_pred, y); epoch_CE_loss += l.item(); CE_loss.append(l.item())
                    avg_CE_loss = epoch_CE_loss / (step +1)  
                    
                    l1_reg.append(sparsity_constraint.item()); epoch_act_map += sparsity_constraint.item()
                    avg_act_map = epoch_act_map / (step +1)

                    reg_ = cfg.train.lambda_l1 * sparsity_constraint; reg.append(reg_.item())     
                    epoch_l1_reg += reg_.item()
                    avg_epoch_l1_reg = epoch_l1_reg / (step +1)       

                loss = loss_function(y_pred, y) + (cfg.train.lambda_l1 * sparsity_constraint) 
            else:
                loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc, avg_bin_acc = estimator.get_accuracy(6)
            if not cfg.data.binary:
                avg_kappa = estimator.get_kappa(6)

            # visualize samples
            if cfg.train.sample_view and step % cfg.train.sample_view_interval == 0:
                samples = torchvision.utils.make_grid(X)
                samples = inverse_normalize(samples, cfg.data.mean, cfg.data.std)
                logger.add_image('input samples', samples, 0, dataformats='CHW')

            if cfg.data.binary:
                list_auc, _, _ = estimator.get_auc_auprc(6)
                progress.set_description(
                    'epoch: [{} / {}], bash: {}/{}, loss: {:.6f}, acc: {:.4f}, auc: {:.4f}'.format(epoch, cfg.train.epochs, step, len(train_loader), avg_loss, avg_acc, list_auc[0]) )
            else:
                progress.set_description(
                    'epoch: [{} / {}], bash: {}/{}, loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}, bin_acc: {:.4f}'
                    .format(epoch, cfg.train.epochs, step, len(train_loader), avg_loss, avg_acc, avg_kappa, avg_bin_acc) )
        
        # Binary metrics: after a complete epoch
        if cfg.data.binary:
            list_auc, list_auprc, list_others = estimator.get_auc_auprc(6)
            t_auc, train_fpr, train_tpr = list_auc
            t_auprc, train_prec_tab, train_sens_tab = list_auprc
            t_sens, t_prec, t_spec, t_cm = list_others

            message = '{} => epoch: [{} / {}], loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, auprc: {:.4f}'
            #progress.set_description(message.format('Training', epoch, cfg.train.epochs, avg_loss, avg_acc, t_auc, t_auprc))
            print(message.format('Training', epoch, cfg.train.epochs, avg_loss, avg_acc, t_auc, t_auprc))

        if cfg.train.sparsity: # batch records
            df2 = pd.DataFrame({'epoch_loss': CE_loss, 'epoch_reg': l1_reg, 'epoch_l1_reg': reg})
            df2.to_csv(os.path.join(save_path, 'summarize_epoch.csv'), index=False)

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            eval(cfg, model, val_loader, estimator, loss_function)
            acc, bin_acc = estimator.get_accuracy(6)
            loss_ = estimator.get_val_loss(6)
            if not cfg.data.binary:
                kappa = estimator.get_kappa(6)
                #print('get kappaaaaaaaaaaaaaaaaaaaaaaaaaaaa', kappa)
            else:
                list_auc, list_auprc, list_others = estimator.get_auc_auprc(6)
                v_auc, val_fpr, val_tpr = list_auc
                v_auprc, val_prec_tab, val_sens_tab = list_auprc
                v_sens, v_prec, v_spec, v_cm = list_others
            
            dtime = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
            if cfg.data.binary:
                message = '{} => epoch {}/{} validation accuracy: {}, auc: {}, auprc: {}'
                progress.set_description(message.format('Validation', epoch, cfg.train.epochs, acc, v_auc, v_auprc))
                print(message.format(dtime, epoch, cfg.train.epochs, acc, v_auc, v_auprc))
            else:
                # message = '{}: epoch {}/{} validation accuracy: {}, kappa: {}, bin acc.: {}'
                #print(message.format(dtime, epoch, cfg.train.epochs, acc, kappa, bin_acc))
                print('{}: epoch {}/{} validation accuracy: {}, kappa: {}'.format(dtime, epoch, cfg.train.epochs, acc, kappa))
            
            if logger:
                logger.add_scalar('validation loss', loss_, epoch)
                logger.add_scalar('validation accuracy', acc, epoch)
            
                if cfg.data.binary:
                    logger.add_scalar('validation AUC', v_auc, epoch)
                    logger.add_scalar('validation AUPRC', v_auprc, epoch)
                    logger.add_scalar('validation sensitivity', v_sens, epoch)
                    logger.add_scalar('validation specificity', v_spec, epoch)
                    logger.add_scalar('validation precision', v_prec, epoch)

                    if cfg.data.num_classes != 2:
                        logger.add_scalar('validation bin accuracy', bin_acc, epoch)
                        val_bin_acc.append(bin_acc)                    

                    fig = matplotlib_roccurve([(train_fpr, train_tpr), (val_fpr, val_tpr)], 
                                            labels=['train', 'val'],
                                            points=points,
                                            point_labels=['WP BDA', 'WP NHS'])
                    logger.add_figure('roc_curves', fig, epoch)

                    prec_sens_fig = matplotlib_prec_recall_curve([(train_sens_tab, train_prec_tab), (val_sens_tab, val_prec_tab)], 
                                            labels=['train', 'val'])
                    logger.add_figure('precision_recall_curves', prec_sens_fig, epoch)

                    cm_fig_valid = plot_conf_matrix(v_cm)
                    logger.add_figure('validation confusion matrix', cm_fig_valid.figure, epoch)

                    val_auc.append(v_auc); val_auprc.append(v_auprc); val_sens.append(v_sens)
                    val_spec.append(v_spec); val_prec.append(v_prec)
                else:
                    logger.add_scalar('validation kappa', kappa, epoch)
                    val_kappa.append(kappa);

            val_loss.append(loss_); val_acc.append(acc)             

            # save model
            (indicator, bin_indicator) = (kappa, bin_kappa) if cfg.train.kappa_prior else (acc, bin_acc)
            if indicator > max_indicator:
                print('save the best model. \n Epoch: {}, acc. {} \n'.format(epoch, indicator))
                #save_weights(model, os.path.join(cfg.base.save_path, 'best_validation_weights_epoch{}_acc_{}.pt'.format(epoch, indicator)))
                save_weights(model, os.path.join(save_path, 'best_validation_weights_acc.pt'))                
                max_indicator = indicator
                #print_msg('Best in validation set. Model save at {}'.format(save_path))

            if (bin_indicator > max_bin_indicator) and (not cfg.data.binary):
                print('save the best model. Epoch: \n {}, bin. acc. {} \n'.format(epoch, bin_indicator))
                #save_weights(model, os.path.join(cfg.base.save_path, 'best_validation_weights_epoch{}_acc_{}.pt'.format(epoch, indicator)))
                save_weights(model, os.path.join(save_path, 'best_validation_weights_bin_acc.pt'))                
                max_bin_indicator = bin_indicator
                #print_msg('Best in validation set. Model save at {}'.format(save_path))

            if min_loss_indicator > loss_:
                print('save the best model based on loss. \n Epoch: {}, loss. {} \n'.format(epoch, loss_))
                save_weights(model, os.path.join(save_path, 'best_validation_weights_loss.pt'))
                min_loss_indicator = loss_
                #print_msg('Best in validation set on the loss. Model save at {}'.format(save_path))

            if cfg.data.binary:
                if v_auc > max_auc:
                    print('save the best model based on the best AUC. \n Epoch: {}, AUC. {} \n'.format(epoch, v_auc))
                    save_weights(model, os.path.join(save_path, 'best_validation_weights_auc.pt'))
                    max_auc = v_auc

                if v_sens > max_sens:
                    print('save the best model based on the best Sensitivity. \n Epoch: {}, Sensitivity. {} \n'.format(epoch, v_sens))
                    save_weights(model, os.path.join(save_path, 'best_validation_weights_sens.pt'))
                    max_sens = v_sens

                if v_spec > max_spe:
                    print('save the best model based on the best Specificity. \n Epoch: {}, Specificity. {} \n'.format(epoch, v_spec))
                    save_weights(model, os.path.join(save_path, 'best_validation_weights_spec.pt'))
                    max_spe = v_spec

                if v_prec > max_pre:
                    print('save the best model based on the best Precision. \n Epoch: {}, Precision. {} \n'.format(epoch, v_prec))
                    save_weights(model, os.path.join(save_path, 'best_validation_weights_prec.pt'))
                    max_pre = v_prec
            else:      
                #print('kapppppppppppppppppppppppppppp', kappa, kappa_indicator)        
                if kappa > kappa_indicator:
                    print('save the best model based on Kappa. Epoch: {}, kappa. {}'.format(epoch, kappa))
                    save_weights(model, os.path.join(save_path, 'best_validation_weights_kappa.pt'))
                    kappa_indicator = kappa
                    print_msg('Best in validation set on the kappa. Model save at {}'.format(save_path))
                    #print('save kapppppppppppppppppppppppppppp') 

        '''
        if epoch % cfg.train.save_interval == 0:
            save_weights(model, os.path.join(cfg.base.save_path, 'epoch_{}.pt'.format(epoch)))
        '''

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if cfg.solver.lr_scheduler == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

            if cfg.data.binary:
                logger.add_scalar('training AUC', t_auc, epoch)
                logger.add_scalar('training AUPRC', t_auprc, epoch)
                logger.add_scalar('training sensitivity', t_sens, epoch)
                logger.add_scalar('training specificity', t_spec, epoch)
                logger.add_scalar('training precision', t_prec, epoch)

                if cfg.data.num_classes != 2:                
                    logger.add_scalar('training bin accuracy', avg_bin_acc, epoch)
                    train_bin_acc.append(avg_bin_acc)

                cm_fig_train = plot_conf_matrix(t_cm)
                logger.add_figure('training confusion matrix', cm_fig_train.figure, epoch)

                train_auc.append(t_auc); train_auprc.append(t_auprc) 
                train_sens.append(t_sens); train_spec.append(t_spec); train_prec.append(t_prec)
            else:
                logger.add_scalar('training kappa', avg_kappa, epoch)
                train_kappa.append(avg_kappa)

            #logger.flush()

        train_loss.append(avg_loss); train_acc.append(avg_acc); 
        lr.append(curr_lr)
        e_CE_loss.append(avg_CE_loss); e_reg.append(avg_act_map); e_l1_reg.append(avg_epoch_l1_reg)

        df = pd.DataFrame({
            'train_loss': train_loss, 'train_acc': train_acc, 'lr': lr, 
            'val_loss': val_loss, 'val_acc': val_acc
            })
            
        #print(f'len list: {len(train_bin_acc), len(train_auc), len(train_bin_kappa), len(val_kappa), len(train_kappa)}')
        if cfg.data.binary:
            if cfg.data.num_classes != 2: 
                df['train_bin_acc'] = train_bin_acc; df['val_bin_acc'] = val_bin_acc
            df['train_auc']= train_auc; df['train_auprc'] = train_auprc; df['train_sensitivity'] = train_sens
            df['train_specificity'] = train_spec; df['train_precision'] = train_prec
            #df['train_bin_kappa'] = train_bin_kappa; #df['val_bin_kappa']= val_bin_kappa

            df['val_auc'] = val_auc;  df['val_auprc'] = val_auprc
            df['val_sensitivity'] = val_sens; df['val_specificity'] = val_spec; df['val_precision'] = val_prec
        else:
             df['train_kappa']= train_kappa
             df['val_kappa'] =  val_kappa
        
        if cfg.train.sparsity:
            df['loss'] = e_CE_loss
            df['l1_reg'] =  e_reg
            df['l1_lambda_reg'] =  e_l1_reg
        
        df.to_csv(os.path.join(save_path, 'summarize.csv'), index=False)

    # save final model
    save_weights(model, os.path.join(save_path, 'final_weights.pt'))
    #save_weights(best_model, os.path.join(cfg.base.save_path, 'final_best_weights.pt'))
    if logger:
        logger.close()   

    mess = 'best validation \n acc= {} \n bin acc. = {} \n auc= {} \n spec= {}, sens= {}, prec= {}, loss= {}'
    print(mess.format(max_indicator, bin_indicator, max_auc, max_spe, max_sens, max_pre, min_loss_indicator)) 

def evaluate(cfg, model, checkpoint, test_dataset, estimator, type_ds):
    weights = torch.load(checkpoint)
    loss = nn.CrossEntropyLoss()
    
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights, strict=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=cfg.train.pin_memory
    )

    print(f'Running on {type_ds} set...')
    eval(cfg, model, test_loader, estimator, loss_func=loss)

    #print('========================================')
    if cfg.data.binary:
        list_auc, list_auprc, list_others = estimator.get_auc_auprc(5)
        auc = list_auc[0]
        auprc, sens, prec, spec = list_auprc[0], list_others[0], list_others[1], list_others[2]

        print('Finished! {} acc {}'.format(type_ds, estimator.get_accuracy(6)[0]))
        print('loss:', estimator.get_val_loss())
        print('AUC: {}, sens: {}, spec: {}, prec: {}, AUPRC: {}'.format(auc, sens, spec, prec, auprc))
        print('Confusion Matrix:')
        print(estimator.conf_mat)   
    else:
        print('acc.: {}'.format(estimator.get_accuracy(6)[0]))
        print('binary acc.: {}'.format(estimator.get_accuracy(6)[1]))
        print('kappa: {}'.format(estimator.get_kappa(6)))
        print(estimator.conf_mat)
    #print('========================================')


def eval(cfg, model, dataloader, estimator, loss_func=None):
    model.eval()
    device = cfg.base.device
    criterion = cfg.train.criterion
    torch.set_grad_enabled(False)
    l = {}
    if loss_func:
        loss_function = loss_func       
        l['op'] = lambda tensor: tensor.item()
    else:
        loss_function = lambda a,b: 1
        l['op'] = lambda a: 0

    estimator.reset()
    epoch_loss, avg_val_loss = 0, 0
    for step, test_data in enumerate(dataloader):
        X, y = test_data
        X, y = X.to(device), y.to(device)
        y = select_target_type(y, criterion)
        #print('y value......: ', y)

        y_pred = model(X)
        '''
        if y.shape[0] == 1:
            y_pred = y_pred.view(-1)
            print('shape 1 prediction shape: ', y_pred.shape)
        '''
        estimator.update(y_pred, y)

        #print('loss: ', loss_func)
        #print('prediction shape: ', y_pred.shape)
        #print('y shape: ', y.shape)
        #print('y shape: ', y.shape[0])

        loss = loss_function(y_pred, y)
        #print('loss: ', loss)
        epoch_loss += l['op'](loss) #loss.item()
        avg_val_loss = epoch_loss / (step + 1)

    #print('valllllllllll')
    #exit()

    if loss_func:
        #print('loss function yes: ', loss_func)
        estimator.update_val_loss(avg_val_loss)
    #else:
    #    print('no loss ', loss_func)

    model.train()
    torch.set_grad_enabled(True)


# define weighted_sampler
def initialize_sampler(cfg, train_dataset):
    sampling_strategy = cfg.data.sampling_strategy
    if sampling_strategy == 'class_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'progressively_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
    else:
        weighted_sampler = None
    return weighted_sampler


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    criterion_args = cfg.criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    nesterov = cfg.solver.nesterov
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.warmup_epochs
    learning_rate = cfg.solver.learning_rate
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
