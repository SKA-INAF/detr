from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.utils.data
import numpy as np
import torchvision
from torchvision.transforms import ToPILImage
import wandb
from torchvision import datasets
import os
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import datasets.transforms as T

class GalaxyDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(GalaxyDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertGalaxyPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(GalaxyDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_galaxy_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class Logger:

    def __init__(self):
        self.CLASSES = ['No-Object', 'Galaxy', 'Source', 'Sidelobe']
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556]]
        


    def log_gt(self, orig_image, targets, out_dir='images', idx=0):
        # no_pad = orig_image[idx] != 0
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)
        target = targets[idx]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(target['boxes'], target['size'])

        # Save image to local file, then re-upload it and convert to PIL
        # This is because plot_results works with plt 
        os.makedirs(out_dir, exist_ok=True)
        out_img_path = f'{out_dir}/out_gt.jpg'

        confidence = [1.0] * target['boxes'].shape[0]
        self.save_output(orig_image, target['labels'], bboxes_scaled, out_img_path, confidence)
        # self.drawBoundingBoxes(orig_image, target['labels'], bboxes_scaled, confidence)

        with Image.open(out_img_path) as im:
            wandb.log({f'GT': wandb.Image(im)})


    def log_predictions(self, orig_image, output, out_dir='images', idx=0):
        # Map the image back to a [0,1] range
        # no_pad = orig_image != 0
        denorm_img, _ = inv_normalize()(orig_image[idx])
        orig_image = torchvision.transforms.ToPILImage()(denorm_img)
        # Take the first image of the batch, discard last logit
        probas = output['pred_logits'].softmax(-1)[idx, :, :-1]
        # keep only predictions with 0.5+ confidence
        keep = probas.max(-1).values > 0.5

        # convert boxes from [0; 1] to image scales
        # Takes the first prediction of the batch, where confidence is higher than 0.5
        bboxes_scaled = self.rescale_bboxes(output['pred_boxes'][0, keep], denorm_img.shape[1:])

        # Save image to local file, then re-upload it and convert to PIL
        # This is because plot_results works with plt 
        os.makedirs(out_dir, exist_ok=True)
        out_img_path = f'{out_dir}/out_pred.jpg'
        # FIXME Implement handling of no prediction
        if keep.any():
            confidence, labels = probas[keep].max(-1)
            self.save_output(orig_image, labels, bboxes_scaled, confidence.tolist())
            # self.drawBoundingBoxes(denorm_img.cpu().numpy(), labels, bboxes_scaled, confidence.tolist())

            with Image.open(out_img_path) as im:
                wandb.log({f'Pred': wandb.Image(im)})

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

    def drawBoundingBoxes(self, pil_img, labels, boxes, confidence):
        """Draw bounding boxes on an image.
        imageData: image data in numpy array format
        imageOutputPath: output image file path
        inferenceResults: inference results array off object (x1,y1,x2,y2)
        colorMap: Bounding box color candidates, list of RGB tuples.
        """
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
            cv2.putText(im_arr_bgr, text, (int(xmin), int(ymin) - 12), 0, 1e-3 * h, c, thickness=3)
            img_arr = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)
            # convert back to Image object
            pil_img = Image.fromarray(img_arr)


            # _, imgHeight, imgWidth = image.shape
            # # thick = int((imgHeight + imgWidth) // 900)
            # thickness = 2
            # text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            # cv2.rectangle(image,(int(xmin), int(ymin)), (int(xmax), int(ymax)), c, thickness)
            # cv2.putText(image, text, (int(xmin), int(ymin) - 12), 0, 1e-3 * imgHeight, c, thickness)
        # cv2.imshow("bounding_box", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def save_output(self, pil_img, labels, boxes, filename, confidence):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        for cl, (xmin, ymin, xmax, ymax), cs, c in zip(labels.tolist(), boxes.tolist(), confidence, colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            text = f'{self.CLASSES[cl]}: {cs:0.2f}'
            
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        # plt.show()
        plt.savefig(filename)

class ConvertGalaxyPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_galaxy_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def inv_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return T.Normalize(
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def make_galaxy_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / 'annotations' / 'train.json'),
        "val": (root / "val", root / 'annotations' / 'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = GalaxyDetection(img_folder, ann_file, transforms=make_galaxy_transforms(image_set), return_masks=args.masks)
    return dataset