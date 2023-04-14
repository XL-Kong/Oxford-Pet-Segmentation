import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["segm"]
    return iou_types


def iou_score(outputs, targets, threshold=0.5):
    """
    Calculates IoU score between outputs and targets
    """
    if not len(outputs):
        return 0.0
    iou_sum = 0.0
    for output, target in zip(outputs, targets):
        intersection = (output & target).float().sum((1, 2))  
        union = (output | target).float().sum((1, 2))
        iou = (intersection + 1e-10) / (union + 1e-10)
        iou = (iou > threshold).float()
        iou_sum += iou.mean().item()
    return iou_sum / len(outputs)

def iou_per_class(outputs, targets, num_classes, threshold=0.5):
    """
    Calculates IoU score for each class with respect to background
    """
    if not len(outputs):
        return 0.0
    iou_scores = [[] for i in range(num_classes)] # create empty list for each class
    for output, target in zip(outputs, targets):
        for c in range(1, 2): # iterate over classes, skip 0 for background
            predicted_masks = output[c] > threshold
            true_masks = target[c]
            iou = iou_score(predicted_masks, true_masks)
            iou_scores[c].append(iou)
    return [sum(iou_scores[c])/len(iou_scores[c]) if len(iou_scores[c])>0 else 0.0 for c in range(num_classes)]


@torch.no_grad()
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    iou_scores_5 = []
    iou_scores_65 = []
    iou_scores_8 = []
    iou_scores_95 = []
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # predicted_masks = outputs["masks"] > 0.5
        # print(len(outputs))
        for i in range(len(outputs)):
            predicted_masks = outputs[i]["masks"] > 0.5
            true_masks = targets[i]["masks"]
            iou_5 = iou_score(predicted_masks, true_masks,threshold=0.5)
            iou_scores_5.append(iou_5)
            iou_65 = iou_score(predicted_masks, true_masks,threshold=0.65)
            iou_scores_65.append(iou_65)
            iou_8 = iou_score(predicted_masks, true_masks,threshold=0.8)
            iou_scores_8.append(iou_8)
            iou_95 = iou_score(predicted_masks, true_masks,threshold=0.95)
            iou_scores_95.append(iou_95)
        
        # for i in range(len(targets)):
        #     iou = iou_score(predicted_masks[i], targets[i])
        #     iou_scores.append(iou)
            
    # print('iou score: ',iou_scores)
    mean_iou_score_5 = np.mean(iou_scores_5)
    mean_iou_score_65 = np.mean(iou_scores_65)
    mean_iou_score_8 = np.mean(iou_scores_8)
    mean_iou_score_95 = np.mean(iou_scores_95)
    print('mIoU(0.5) score: ',mean_iou_score_5)
    print('mIoU(0.95) score: ',mean_iou_score_95)
    print('mIoU(0.5) score: ',mean_iou_score_5)
    print('mIoU(0.5) score: ',mean_iou_score_5)
    return mean_iou_score_5


# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     print(data_loader.dataset)
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     print(iou_types)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for image, targets in metric_logger.log_every(data_loader, 100, header):
#         image = list(img.to(device) for img in image)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(image)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         # model_time = time.time() - model_time
#         iou_score(outputs, targets)        
#         # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         # evaluator_time = time.time()
#         # coco_evaluator.update(res)
#         # evaluator_time = time.time() - evaluator_time
#         # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # # gather the stats from all processes
#     # metric_logger.synchronize_between_processes()
#     # print("Averaged stats:", metric_logger)
#     # coco_evaluator.synchronize_between_processes()

#     # # accumulate predictions from all images
#     # coco_evaluator.accumulate()
#     # coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator

