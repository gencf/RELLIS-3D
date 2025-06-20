# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist
from PIL import Image


def convert_label(label, inverse=False):
    label_mapping = {0: 0,
                     1: 0,
                     3: 1,
                     4: 2,
                     5: 3,
                     6: 4,
                     7: 5,
                     8: 6,
                     9: 7,
                     10: 8,
                     12: 9,
                     15: 10,
                     17: 11,
                     18: 12,
                     19: 13,
                     23: 14,
                     27: 15,
                    #  29: 1,
                    #  30: 1,
                     31: 16,
                    #  32: 4,
                     33: 17,
                     34: 18}
    temp = label.copy()
    if inverse:
        for v,k in label_mapping.items():
            temp[label == k] = v
    else:
        for k, v in label_mapping.items():
            temp[label == k] = v
    return temp

def convert_color(label, color_map):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
        for k,v in color_map.items():
            temp[label == k] = v
        return temp
    

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )
                
            if idx == 0:
                print("Validate: [{}/{}]".format(idx, len(testloader)))

            if (idx+1) % 10 == 0:
                print("Validate: [{}/{}]".format((idx+1), len(testloader)))

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # print(IoU_array)
        mean_IoU = IoU_array.mean()
        # if dist.get_rank() <= 0:
        #     logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False, viz=True, id_color_map={}):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)
            pred = convert_label(pred, True)
            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True, viz=True, id_color_map={}):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = model(image)
            pred = pred[config.TEST.OUTPUT_INDEX]
            pred_np = pred.cpu().numpy()
            b,_,_,_ = pred.shape
            for i in range(b):
                sv_path = os.path.join(config.OUTPUT_DIR, 'hrnet',name[i],'pylon_camera_node_label_id')
                if not os.path.exists(sv_path):
                    os.makedirs(sv_path)
                _, file_name = os.path.split(name[i])
                file_name = file_name.replace("jpg","png")
                data_path = os.path.join(sv_path,file_name)
                pred_arg = np.argmax(pred_np[i],axis=0).astype(np.uint8)
                pred_arg = convert_label(pred_arg, True)
                pred_img = np.stack((pred_arg,pred_arg,pred_arg),axis=2)
                pred_img = Image.fromarray(pred_img)
                pred_img.save(data_path)
                if viz:
                    sv_path = os.path.join(config.OUTPUT_DIR, 'hrnet',name[i],'pylon_camera_node_label_color')
                    if not os.path.exists(sv_path):
                        os.makedirs(sv_path)
                    _, file_name = os.path.split(name[i])
                    file_name = file_name.replace("jpg","png")
                    color_path = os.path.join(sv_path,file_name)
                    color_label = convert_color(pred_arg, id_color_map)
                    # pred_arg_flat = np.ravel(pred_arg)
                    # print(pred_arg)
                    # color_arg = np.array([id_color_map.get(i,(i,i,i)) for i in pred_arg_flat]).astype(np.uint8)
                    # pred_arg_back = np.reshape(color_arg, (pred_arg.shape[0],pred_arg.shape[1],-1))
                    # print(pred_arg_back)
                    # print(pred_arg_back[:,:,0])
                    # color_img = Image.fromarray(np.reshape(color_arg,(pred_arg.shape[0],pred_arg.shape[1],-1)))
                    color_img = Image.fromarray(color_label,'RGB')
                    color_img.save(color_path)
                #np.save(data_path,pred_arg)
            # pred = test_dataset.multi_scale_inference(
            #     config,
            #     model,
            #     image,
            #     scales=config.TEST.SCALE_LIST,
            #     flip=config.TEST.FLIP_TEST)

            # if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
            #     pred = F.interpolate(
            #         pred, size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )

            # if sv_pred:
            #     sv_path = os.path.join(sv_dir, 'test_results')
            #     if not os.path.exists(sv_path):
            #         os.mkdir(sv_path)
            #     test_dataset.save_pred(pred, sv_path, name)