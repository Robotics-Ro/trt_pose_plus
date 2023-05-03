# --------------------------------------------------------
# Camera Recorder for Tegra X2/X1
#
# This program captures video from IP CAM, USB webcam,
# or the Tegra onboard camera, adds some watermark on
# the video frames and then records it into a TS file.
# The code demonstrates how to use cv2.VideoWriter()
# while taking advantage of TX2/TX1's H.264 H/W encoder
# capabilities.
#
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-recorder/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------


import sys
import argparse
import subprocess

import cv2


def parse_args():
    # Parse input arguments
    desc = 'Capture and record live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [640]',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [480]',
                        default=480, type=int)
    parser.add_argument('--ts', dest='ts_file',
                        help='output .ts file name ["output"]',
                        default='output', type=str)
    parser.add_argument('--fps', dest='fps',
                        help='fps of the output .ts video [30]',
                        default=30, type=int)
    parser.add_argument('--rec', dest='rec_sec',
                        help='recording length in seconds [5]',
                        default=5, type=int)
    parser.add_argument('--text', dest='text',
                        help='text overlayed on the video ["TX2 DEMO"]',
                        default='TX2 DEMO', type=str)
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def get_video_writer(fname, fps, width, height):
    gst_str = ('appsrc ! videoconvert ! omxh264enc ! mpegtsmux ! '
               'filesink location={}.ts').format(fname)
    print(gst_str)
    return cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, fps,
                           (width, height))


def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)
    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    writer = get_video_writer(args.ts_file, args.fps,
                              args.image_width, args.image_height)
    if not writer.isOpened():
        cap.release()
        sys.exit('Failed to open "{}.ts"!'.format(args.ts_file))

    for _ in range(args.fps * args.rec_sec): # assuming python3 here...
        _, img = cap.read() # read image
        if img is None:
            print('No more image from camera, exiting...')
            break
        # put watermark on image
        cv2.putText(img, args.text, (10, 60),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2.0,
                    color=(0, 240, 0), thickness=6, lineType=cv2.LINE_AA)
        cv2.imshow('REC', img)
        if cv2.waitKey(1) == 27: # ESC key: quit program
            break
        writer.write(img) # write current image to the ts file

    writer.release()
    cap.release()


if __name__ == '__main__':
    main()