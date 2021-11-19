from PIL import Image, ImageDraw, ImageFont
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import json
import wandb
import os
import util.misc as utils
import torchvision
import numpy as np
import torch
from datasets.galaxy import inv_normalize

class Logger:

    def __init__(self):
        self.CLASSES = ['No-Object', 'Galaxy', 'Source', 'Sidelobe']
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556]]
        # self.COLORS = {'No-Object': (0, 114, 189), 'Galaxy': (217, 83, 25), 'Source': (237, 177, 32), 'Sidelobe': (126, 47, 142)}
        # self.COLORS = [(0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142)]
        # alpha = 128
        # self.COLORS_RGBA = [(0, 114, 189, alpha), (217, 83, 25, alpha), (237, 177, 32, alpha), (126, 47, 142, alpha)]
        
    def log_gt(self, orig_image, targets, idx=0):
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)
        target = targets[idx]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(target['boxes'], target['size'])

        confidence = [1.0] * target['boxes'].shape[0]
        if 'masks' in target.keys():
            self.log_image_w_mask(orig_image, target['labels'], bboxes_scaled, target['masks'], confidence, 'Ground Truth')
        else:
            self.log_image(orig_image, target['labels'], bboxes_scaled, confidence, 'Ground Truth')



    def log_predictions(self, orig_image, output, idx=0):
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)

        # Take the first image of the batch, discard last logit
        pred_logits = output['pred_logits'].softmax(-1)[idx, :, :-1]
        pred_boxes = output['pred_boxes'][idx]
        
        pred_masks = None
        if 'pred_masks' in output.keys():
            pred_masks = output['pred_masks'][idx]
        
        class_probs = pred_logits.max(-1)
        # keep only predictions with 0.5+ confidence
        keep = class_probs.values > 0.7

        # Save image to local file, then re-upload it and convert to PIL
        # FIXME Implement handling of no prediction
        if keep.any():
            pred_logits = pred_logits[keep]
            pred_boxes = pred_boxes[keep]

            # convert boxes from [0; 1] to image scales
            # Takes the first prediction of the batch, where confidence is higher than 0.5
            bboxes_scaled = self.rescale_bboxes(pred_boxes, denorm_img.shape[1:])

            confidence, labels = class_probs
            confidence = confidence[keep]
            labels = labels[keep]

            if pred_masks:
                pred_masks = pred_masks[keep]
                self.log_image_w_masks(orig_image, labels, bboxes_scaled, pred_masks, confidence.tolist(), 'Prediction')

            else:
                self.log_image(orig_image, labels, bboxes_scaled, confidence.tolist(), 'Prediction')
    
    def save_bboxes(self, targets, output, out_file, idx=0):
        pred_logits = output['pred_logits'].softmax(-1)[idx, :, :-1]
        pred_boxes = output['pred_boxes'][idx]
        img_id = targets[idx]['image_id']
        
        class_probs = pred_logits.max(-1)
        keep = class_probs.values > 0.5
        
        if keep.any():
            # TODO Check if enters condition 
            pred_boxes = pred_boxes[keep]

        img_to_boxes = {img_id.item(): pred_boxes.tolist()}
        json.dump(img_to_boxes, out_file)
            

    # for output bounding box post-processing
    def box_xywh_to_xyxy(self, x):
        '''
        Converts bounding box coordinates 
        from the form [x0, y0, w, h] to [x0, y0, x1, y1]
        '''

        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        # img_w, img_h = size
        img_h, img_w = size
        b = self.box_xywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
        return b

    def log_image(self, pil_img, labels, boxes, confidence, title):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for cl, (xmin, ymin, xmax, ymax), cs, in zip(labels.tolist(), boxes.tolist(), confidence):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=self.COLORS[cl], linewidth=3))
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        if utils.is_main_process():
            wandb.log({f'{title}': wandb.Image(ax)})

    def log_image_w_mask(self, pil_img, labels, boxes, masks, confidence, title):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for cl, (xmin, ymin, xmax, ymax), m, cs, in zip(labels.tolist(), boxes.tolist(), masks, confidence):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=self.COLORS[cl], linewidth=3))
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
            
            # Swap dimensions to take x points as firsts
            # [H, W] => [W, H] 
            m = m.t()
            # Get mask coordinates
            poly = (m == True).nonzero()
            if len(poly):
                ax.add_patch(Polygon(poly.cpu(), color=self.COLORS[cl], alpha=0.7))
        if utils.is_main_process():
            wandb.log({f'{title}': wandb.Image(ax)})

    def log_PIL_image(self, pil_img, labels, boxes, masks, confidence, title):
        im = pil_img.copy()
        drw = ImageDraw.Draw(im, 'RGBA')
        for cl, (xmin, ymin, xmax, ymax), m, cs, in zip(labels.tolist(), boxes.tolist(), masks, confidence):
            drw.rectangle([xmin, ymin, xmax, ymax], outline=self.COLORS[cl], width=3)
            font = ImageFont.load_default()
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            fw, fh = font.getsize(text)
            drw.rectangle([xmin, ymin - fh, xmin + fw, ymin], fill=self.COLORS[cl])
            
            # Swap dimensions to take x points as firsts
            # [H, W] => [W, H] 
            m = m.t()
            # Get mask coordinates
            poly = (m == True).nonzero()
            # Convert lists to tuples as ImageDraw accepts them this way
            mask_points = list(map(lambda x: tuple(x), poly.tolist()))
            if mask_points:
                drw.polygon(mask_points, fill=self.COLORS_RGBA[cl])
                drw.text((xmin, ymin - 12), text, fill=(255,255,255), font=font)
        if utils.is_main_process():
            wandb.log({f'{title}': wandb.Image(im)})


class FFRLogger(Logger):
    def __init__(self):
        super(FFRLogger, self).__init__()
        self.CLASSES = ['No-Object', 'Stenosis']
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]