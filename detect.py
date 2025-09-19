"""
Inference script for CTD-YOLO
Run inference on images, videos, directories, streams, etc.
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.ctd_yolo import CTD_YOLO
from utils import (check_img_size, non_max_suppression, scale_coords, 
                   xyxy2xywh, plot_one_box, colorstr, increment_path)


@torch.no_grad()
def run(weights='runs/train/exp/weights/best.pt',
        source='data/images',
        data='data/ce-cctsdb.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
        use_cdm=False,
        cdm_weights='',
        ):
    """
    Run inference with CTD-YOLO model
    """
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = CTD_YOLO(weights, device=device, dnn=dnn, data=data, fp16=half, use_cdm=use_cdm)
    
    # Load CDM pretrained weights
    if use_cdm and cdm_weights and Path(cdm_weights).exists():
        print(f'Loading CDM weights from {cdm_weights}')
        cdm_ckpt = torch.load(cdm_weights, map_location='cpu')
        if hasattr(model, 'cdm'):
            model.cdm.load_state_dict(cdm_ckpt['model_state_dict'], strict=False)
            print('CDM weights loaded successfully')
    
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/ce-cctsdb.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--use-cdm', action='store_true', help='use CDM preprocessing')
    parser.add_argument('--cdm-weights', type=str, default='', help='CDM pretrained weights path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


# Helper functions and classes
def select_device(device='', batch_size=0, newline=True):
    """Select device for inference"""
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s = f"CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"
    else:
        s = 'CPU'

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')


def check_file(file):
    """Search/download file and return path"""
    file = str(file)  # convert to str()
    if Path(file).is_file() or not file:  # exists
        return file
    elif file.startswith(('http://', 'https://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_imshow(warn=False):
    """Check if environment supports image displays"""
    try:
        assert not any((os.environ.get(x) for x in ('COLAB_GPU', 'KAGGLE_URL_BASE')))  # in colab/kaggle
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def print_args(args: dict):
    """Print function arguments"""
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in args.items()))


def check_requirements(requirements=(), exclude=(), install=True):
    """Check installed packages"""
    pass


def strip_optimizer(f='best.pt', s=''):
    """Strip optimizer from saved model"""
    pass


class Profile:
    """Profiling context manager"""
    def __init__(self):
        self.dt = 0.0
        self.t = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = time.time() - self.start
        self.t += self.dt


class Annotator:
    """Annotator for drawing bounding boxes"""
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label"""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                       thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        """Return annotated image"""
        return self.im


def colors(i, bgr=False):
    """Colors palette"""
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = [tuple(int(hex[i % len(hex)][j:j + 2], 16) for j in (0, 2, 4)) for i in range(len(hex))]
    n = len(palette)
    c = palette[int(i) % n]
    return (c[2], c[1], c[0]) if bgr else c


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels"""
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
    return crop


# Placeholder classes for data loading
class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True, vid_stride=1):
        self.mode = 'image'
        self.files = [path]
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Simplified implementation
        return 'path', np.zeros((640, 640, 3)), np.zeros((640, 640, 3)), None, ''


class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True, vid_stride=1):
        self.mode = 'stream'
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return ['path'], [np.zeros((640, 640, 3))], [np.zeros((640, 640, 3))], None, ''


class LoadScreenshots:
    def __init__(self, source, img_size=640, stride=32, auto=True):
        self.mode = 'screen'
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return 'screen', np.zeros((640, 640, 3)), np.zeros((640, 640, 3)), None, ''


# Constants
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

# Import required modules
import glob
import urllib.parse
import numpy as np
from utils import clip_coords, xywh2xyxy
