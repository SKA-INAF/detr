import torch
import pandas as pd
import torchvision.transforms.functional as TF
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T 

from pathlib import Path, PurePath

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def rescale_bboxes(out_bbox, size):
        # img_w, img_h = size
        img_h, img_w = size
        b = box_xcycwh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
        return b

def box_xcycwh_to_xyxy(x):
        '''
        Converts bounding box coordinates 
        from the form [x0, y0, w, h] to [x0, y0, x1, y1]
        '''

        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


def inv_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return TF.normalize(
        x,
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def log_gt(orig_image, targets, idx=0):
    denorm_img = inv_normalize()(orig_image[idx])
    orig_image = TF.to_pil_image(denorm_img)
    target = targets[idx]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(target['boxes'], target['size'])

    confidence = [1.0] * target['boxes'].shape[0]
    log_image(orig_image, target['labels'], bboxes_scaled, confidence, 'Ground Truth')


def format_output(output, confidence_threshold=0.7, batch_idx=0):
    '''
    output = output of the model forward step
    img_shape: (w, h)
    '''
    # denorm_img = inv_normalize()(orig_image[idx])
    # orig_image = TF.to_pil_image(denorm_img)

    # Take the first image of the batch, discard last logit
    pred_logits = output['pred_logits'].softmax(-1)[batch_idx, :, :-1]
    pred_boxes = output['pred_boxes'][batch_idx]

    class_probs = pred_logits.max(-1)
    keep = class_probs.values > confidence_threshold

    if keep.any():
        pred_logits = pred_logits[keep]
        pred_boxes = pred_boxes[keep]

        confidence, labels = class_probs
        confidence = confidence[keep]
        labels = labels[keep]

    else:
        labels = None
        pred_boxes = None
        confidence = None


    return labels, pred_boxes, confidence


def log_image(pil_img, labels, boxes, confidence, title, CLASSES, COLORS):

    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.tolist()

    try:
        x = 2
        for cl, (xmin, ymin, xmax, ymax), cs, in zip(labels, boxes, confidence):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl], linewidth=3))
            text = f'{CLASSES[cl]}: {cs:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    finally:
        plt.axis('off')
        return fig
        
def log_image_rev(pil_img, labels, boxes, confidence, title, CLASSES, COLORS):

    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.tolist()

    try:
        for cl, (xmin, ymin, xmax, ymax), cs, in zip(labels, boxes, confidence):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl], linewidth=3))
            text = f'{CLASSES[cl]}: {cs:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    finally:
        plt.axis('off')
        return fig

def apply_transforms(img):
    im_tensor = TF.to_tensor(img)
    return TF.normalize(im_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])