from PIL import Image, ImageDraw, ImageFont
import cv2
import wandb
import os
import torchvision
import numpy as np
import torch
from datasets.galaxy import inv_normalize

class Logger:

    def __init__(self):
        self.CLASSES = ['No-Object', 'Galaxy', 'Source', 'Sidelobe']
        # self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #   [0.494, 0.184, 0.556]]
        self.COLORS = [(0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142)]
        
    def log_gt(self, orig_image, targets, idx=0):
        # no_pad = orig_image[idx] != 0
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)
        target = targets[idx]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(target['boxes'], target['size'])

        # Save image to local file, then re-upload it and convert to PIL
        # This is because plot_results works with plt 

        confidence = [1.0] * target['boxes'].shape[0]
        # self.save_output(orig_image, target['labels'], bboxes_scaled, out_img_path, confidence)
        self.log_image(orig_image, target['labels'], bboxes_scaled, confidence, 'Ground Truth')
        # self.drawBoundingBoxes(orig_image, target['labels'], bboxes_scaled, confidence)


    def log_predictions(self, orig_image, output, idx=0):
        # Map the image back to a [0,1] range
        # no_pad = orig_image != 0
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)
        # Take the first image of the batch, discard last logit
        pred_logits = output['pred_logits'].softmax(-1)[idx, :, :-1]
        pred_boxes = pred_boxes=output['pred_boxes'][idx]
        class_probs = pred_logits.softmax(-1).max(-1)
        # keep only predictions with 0.5+ confidence
        keep = class_probs.values > 0.5

        # Take the best k predictions
        # topk = class_probs.values.topk(15)
        # pred_logits = pred_logits[topk.indices]
        # pred_boxes = pred_boxes[topk.indices]

        # convert boxes from [0; 1] to image scales
        # Takes the first prediction of the batch, where confidence is higher than 0.5
        bboxes_scaled = self.rescale_bboxes(pred_boxes, denorm_img.shape[1:])

        # Save image to local file, then re-upload it and convert to PIL
        # FIXME Implement handling of no prediction
        if keep.any():
            confidence, labels = class_probs[keep].max(-1)
            self.log_image(orig_image, labels, bboxes_scaled, confidence.tolist(), 'Prediction')
            # self.drawBoundingBoxes(denorm_img.cpu().numpy(), labels, bboxes_scaled, confidence.tolist())

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
        return b

    def log_image(self, pil_img, labels, boxes, confidence, title):
        im = pil_img.copy()
        drw = ImageDraw.Draw(im)
        colors = self.COLORS * 100
        for cl, (xmin, ymin, xmax, ymax), cs, c in zip(labels.tolist(), boxes.tolist(), confidence, colors):
            drw.rectangle([xmin, ymin, xmax, ymax], outline=c, width=3)
            font = ImageFont.truetype('arial')
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            fw, fh = font.getsize(text)
            drw.rectangle([xmin, ymin - fh, xmin + fw, ymin], fill=c)
            drw.text((xmin, ymin - 12), text, fill=(55,55,55), font=font)
        wandb.log({f'{title}': wandb.Image(im)})

    def drawBoundingBoxes(self, pil_img, labels, boxes, confidence):

        colors = self.COLORS * 100
        # im is a PIL Image object
        w, h = pil_img.size
        for cl, (xmin, ymin, xmax, ymax), cs, c in zip(labels.tolist(), boxes.tolist(), confidence, colors):

            img_arr = np.asarray(pil_img)
            # convert rgb array to opencv's bgr format
            im_arr_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            # pts1 and pts2 are the upper left and bottom right coordinates of the rectangle
            cv2.rectangle(im_arr_bgr,(int(xmin), int(ymin)), (int(xmax), int(ymax)), c, thickness=3)
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            cv2.putText(im_arr_bgr, text, (int(xmin), int(ymin) - 12), 0, 1e-3 * h, c, thickness=1)
            img_arr = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)
            # convert back to Image object
            pil_img = Image.fromarray(img_arr)
        wandb.log({'Image': wandb.Image(pil_img)})
