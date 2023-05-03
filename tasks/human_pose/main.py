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

import glob
from DetectorLoader import TinyYOLOv3_onecls

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
  
source = 0

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

def execute(img, src, t):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                #plus code by thro 20230501
                

                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.circle(src, (x, y), 3, color, 2)

               
    #print("FPS:%f "%(fps))
    draw_objects(src, counts, objects, peaks)

    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #out_video.write(src)
    return src

parser = argparse.ArgumentParser(description='TensorRT pose estimation run and detect by YOLOv3 Tiny')
parser.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
parser.add_argument('--detection_input_size', type=int, default=384,
                    help='Size of input in detection model in square must be divisible by 32 (int).')
parser.add_argument('--pose_input_size', type=str, default='224x160',
                    help='Size of input in pose model must be divisible by 32 (h, w)')
parser.add_argument('--pose_backbone', type=str, default='resnet50',
                    help='Backbone model for SPPE FastPose model.')
parser.add_argument('--show_detected', default=False, action='store_true',
                    help='Show all bounding box from detection.')
parser.add_argument('--show_skeleton', default=True, action='store_true',
                    help='Show skeleton pose.')
parser.add_argument('--save_out', type=str, default='',
                    help='Save display to video file.')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to run model on cpu or cuda.')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

device = args.device

# DETECTION MODEL.
inp_dets = args.detection_input_size
detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

with open('human_pose.json', 'r') as f:
    
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print(num_parts, num_links)

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
device_torch = torch.device(device)

def preprocess(image):
    global device_torch
    device_torch = torch.device(device)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device_torch)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

count = 0



cap = cv2.VideoCapture("v4l2src device=/dev/video3 ! video/x-raw, width=640, height=480, format=(string)YUY2 ! xvimagesink -e")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

outvid = False
if args.save_out != '':
    outvid = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    gst_out = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video  /x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=test.mkv"
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(fps), (inp_dets * 2, inp_dets * 2))
    # movie = cv2.VideoWriter()

# ret_val, img = cap.read() 
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out_video = cv2.VideoWriter('/Movie/output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))


X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

while cap.isOpened() and True:
    t = time.time()
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break

    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    src = execute(img, dst, t)

    if outvid:
        writer.write(src)
            # cv2.imwrite("./Movie/%06d.jpg" % count, src)
            # print('Read a new frame: ', ret_val)
            # if count == 200 :
            #     break
            # count +=1

            # frames = []
            # imgs = glob.glob("./Movie/*.jpg")
            # for i in imgs :
            #     new_frame = PIL.Image(i)
            #     frames.append(new_frame)

            # frames[0].save('/Movie/result.gif', format='GIF',
            #                 apped_images=frames[1:],
            #                 save_all=True,
            #                 duration=0.2,
            #                 loop=0
            # )

    cv2.imshow('src', src)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #count += 1
if outvid:
    writer.release()

cv2.destroyAllWindows()
# out_video.release()
cap.release()
