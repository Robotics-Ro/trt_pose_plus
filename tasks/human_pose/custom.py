# main.py

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path

## 20230503 THRO - Attend python Library
from Detection.Utils import ResizePadding
from fn import draw_single
import numpy as np 
from DetectorLoader import TinyYOLOv3_onecls
from CameraLoader import CamLoader, CamLoader_Q
from Track.Tracker import Detection, Tracker

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''

source_dir = '/Test'
source = 'test.avi'

def get_keypoint(humans, hnum, peaks):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
    

def kpt2bbox(keypoints):
    # keypoints: numpy array of shape (17, 3), where each row is (x, y, score)

    x = keypoints[:, 0]
    y = keypoints[:, 1]
    scores = keypoints[:, 2]

    # get bounding box coordinates
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    # calculate bounding box size
    width = x_max - x_min
    height = y_max - y_min

    # calculate center point
    center_x = x_min + width / 2
    center_y = y_min + height / 2

    # calculate confidence score
    score = np.mean(scores)

    # return as bbox array
    return np.array([center_x, center_y, width, height, score])

def execute(img, src, get_time):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    
    # Detect humans bbox in the frame with detector model.
    detected = detect_model.detect(img, need_resize=False, expand_bb=5)

    # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
    tracker.predict()
    # Merge two source of predicted bbox together.
    for track in tracker.tracks:
        det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        detected = torch.cat([detected, det], dim=0) if detected is not None else det

    detections = []  # List of Detections object for tracking.

    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)

                # Create Detections object.
                bbox = kpt2bbox(keypoints)
                detections = [Detection(bbox, keypoints, np.mean(keypoints[:, 2]))]

                cv2.circle(src, (x, y), 3, color, 2)
                frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
                frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - get_time)),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frame = frame[:, :, ::-1]
                get_time = time.time()
                cv2.circle(src, (x, y), 3, color, 2)

                # VISUALIZE.
            if args.show_detected:
                for i, (kpt, bbox, score) in enumerate(detections):
                    if score > 0.5:
                        # Visualize keypoints
                        kpt = kpt[:, :2]
                        src = cv2.rectangle(src, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        for j in range(kpt.shape[0]):
                            x, y = int(kpt[j][0]), int(kpt[j][1])
                            cv2.circle(src, (x, y), 3, (0, 0, 255), -1)
    if detected is not None:  
        for box in detected[:, 0:5]:
            # Extract box coordinates.
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Extract the box's image patch.
            patch = frame[y1:y2, x1:x2, :]
            
            # # Detect keypoints in the patch.
            # keypoints = keypoint_detector.detect(patch)
            
            # Convert keypoints to absolute coordinates.
            keypoints[:, 0] += x1
            keypoints[:, 1] += y1
            
            # Create Detection object and add to list.
            detection = Detection((x1, y1, x2, y2), keypoints, None)
            detections.append(detection)

            for bb in detected[:, 0:4]:
                x1, y1, x2, y2 = bb
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Update tracks by matching each track information of current and previous frame or
    # create a new track if no matched.
    tracker.update(detections)
          
    #print("FPS:%f "%(fps))
    draw_objects(src, counts, objects, peaks)
    return src

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='TensorRT pose estimation run and YOLOv3 Tiny Detection')
    par.add_argument('-C', '--camera', default=source_dir + source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x224',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    par.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    #region trt_pose pytorct to TensorRT
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    if 'resnet' in args.model:
        print('------ model = resnet--------')
        MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        WIDTH = 224
        HEIGHT = 224

    else:    
        print('------ model = densenet--------')
        MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
        OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
        WIDTH = 256
        HEIGHT = 256

    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    if os.path.exists(OPTIMIZED_MODEL) == False:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    t0 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(50):
        y = model_trt(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    print(50.0 / (t1 - t0))

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

    #endregion

    resize_fn = ResizePadding(inp_dets, inp_dets)
    
    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        img = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        src = execute(img, frame, cam.get_time)

        cv2.imshow('src', src)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if outvid:
            writer.release()
        

    X_compress = 640.0 / WIDTH * 1.0
    Y_compress = 480.0 / HEIGHT * 1.0
    # parse_objects = ParseObjects(topology)
    # draw_objects = DrawObjects(topology)

    cam.stop()
    cv2.destroyAllWindows()
