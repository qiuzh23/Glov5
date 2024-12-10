# glov5/detect.py
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import torchvision
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]           # 获取当前文件所在这一级的父目录

# 配置参数
WEIGHTS = str(ROOT / 'balloon60.pt')
IMG_SIZE = [640, 640]
CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_DET = 1000
DEVICE = 'cpu'  # 根据需求修改为 'cuda' 如果有GPU支持

def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].to(device).float().eval()
    return model

def preprocess_image(img, img_size, stride=32):
    h, w = img.shape[:2]
    ratio = min(img_size[0] / h, img_size[1] / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    pad_w = (img_size[1] - new_w) // 2
    pad_h = (img_size[0] - new_h) // 2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, img_size[0] - new_h - pad_h,
        pad_w, img_size[1] - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_padded, img

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_boxes(img_shape, boxes, original_shape, gain=None, pad=None):
    if gain is None:
        gain = min(img_shape[0] / original_shape[0], img_shape[1] / original_shape[1])
    if pad is None:
        pad = ((img_shape[1] - original_shape[1] * gain) / 2, (img_shape[0] - original_shape[0] * gain) / 2)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes.clamp_(min=0, max=torch.tensor(original_shape[::-1]))
    return boxes

def non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45, max_det=1000):
    bs, _, nc = pred.shape
    conf = pred[..., 4]
    mask = conf > conf_thresh
    pred = pred[mask]
    if not pred.size(0):
        return [torch.zeros((0, 6), device=pred.device)]
    
    pred[:, 5:] *= conf.unsqueeze(1)
    box = xywh2xyxy(pred[:, :4])
    scores, cls = pred[:, 5:].max(1, keepdim=True)
    detections = torch.cat((box, scores, cls.float()), 1)
    detections = detections[scores.view(-1) > conf_thresh]
    
    if not detections.size(0):
        return [torch.zeros((0, 6), device=pred.device)]
    
    detections = detections[detections[:, 4].argsort(descending=True)]
    keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thresh)
    return [detections[keep[:max_det]]]

def run_inference(model, img, img_size, device, conf_thresh, iou_thresh, max_det):
    img_prep, original_img = preprocess_image(img, img_size)
    img_tensor = torch.from_numpy(img_prep.transpose(2, 0, 1)).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0) if img_tensor.ndimension() == 3 else img_tensor
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, max_det)
    
    window = [960, 540, 320, 180]  # 默认窗口
    for det in pred:
        if det.size(0):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_img.shape)
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = xyxy
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                window = [cx.item(), cy.item(), w.item(), h.item()]
                label = f'{model.names[int(cls)]} {conf:.2f}'
                print(f"Detected: {label}, Window: {window}")
    return window

def get_frame():
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    _, data = client.GetImageSample()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image

def main(model):
    img = get_frame()
    window = run_inference(model, img, IMG_SIZE, DEVICE, CONF_THRESH, IOU_THRESH, MAX_DET)
    cv2.destroyAllWindows()
    return window

if __name__ == "__main__":
    ChannelFactoryInitialize(0)
    model = load_model(WEIGHTS, DEVICE)
    while True:
        main(model)
