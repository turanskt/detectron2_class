#Standard imports
import argparse, random, os, os.path, threading, time, io, pdb
from datetime import datetime
from sys import getsizeof

#Machine learning imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from imgcls.modeling import ClsNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Image processing imports
import cv2

#Gstreamer imports
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, GstBase, Gtk

#CAPS (video properties) definition    
vid_width = 640
vid_height = 480
pred_width = 224
pred_height = 224
FPS = 20/1
ISBOARDSTART = True

#Gstreamer Elements
HHH = ":qtdemux ! queue ! h264parse ! avdec_h264"
SRC_VID = "filesrc location=/home/pi/yoli/yoli.mp4"
SRC_CAM= "v4l2src device=/dev/video0 ! videoscale"
H264_ENC = "qtdemux ! queue ! h264parse ! avdec_h264 ! videoconvert"
CAPS_VID =f"videoscale ! video/x-raw,width={vid_width},height={vid_height},framerate=30/1"
TEXT_OVERLAY = "textoverlay name=oover"
JPEG_ENC = "jpegenc ! rtpjpegpay"
UDP_SINK = "udpsink host=10.15.15.16 port=5000"
CAPS_PRED = f"videoscale ! video/x-raw,width={pred_width},height={pred_height}"
RGB_CONV = "videoconvert ! video/x-raw,format=RGB ! videoconvert"
MAIN_SINK = "appsink name=main-sink emit-signals=true max-buffers=1 drop=true"

pipeline = f"""{SRC_VID} ! {H264_ENC} ! {CAPS_VID} ! tee name=t ! queue ! {TEXT_OVERLAY} ! {JPEG_ENC} ! {UDP_SINK}
    t. ! queue !  {CAPS_PRED} ! {RGB_CONV} ! {MAIN_SINK}
    """

#pipeline = f"""{SRC_CAM} ! {CAPS_VID} ! tee name=t ! queue ! {TEXT_OVERLAY} ! {JPEG_ENC} ! {UDP_SINK}
#    t. ! queue !  {CAPS_PRED} ! {RGB_CONV} ! {MAIN_SINK}
#    """

#Folder check and path set
predictor_path = os.getcwd()
if not os.path.exists("video_output"):
    os.system("mkdir video_output")
video_output_path = f"{predictor_path}/video_output"
if not os.path.exists(f"{video_output_path}/sync"):
    os.system(f"mkdir {video_output_path}/sync")
sync_path = f"{video_output_path}/sync"


Gst.init(None)


class Motion:

    def __init__(self):
        self.Nb_Frames = 0
        self.MIN_CNT = 15000
        self.MAX_CNT = 20000
        self.THRESHOLD = 0.1
        self.b = FrameBuffer()

    def hasContour(self, image):
        #Detect if there is a certain amount of contour given a min and max nb.
        threshold = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations = 2)
        cnts, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            contourArea = cv2.contourArea(contour)
            if contourArea > self.MIN_CNT and contourArea < self.MAX_CNT:
                print(contourArea)
                return True

    def motionMatrix(self, buffer, next_frame):
        #Compare next_frame to all the frames contained inside a buffer.
        ismotionmatrix = []
        for previous_frame in buffer:
            diff = cv2.absdiff(previous_frame,next_frame)
            if self.hasContour(diff):
                ismotionmatrix.append(1)
            else :
                ismotionmatrix.append(0)
        return ismotionmatrix
    

    def hasMotion(self, frame):
        #xxx
        hasmotion = False
        buffer = self.b.appendFrame(frame)

        if self.b.bufferReady() and self.Nb_Frames % FPS == 0: 
            motionmatrix = self.motionMatrix(buffer, self.b.currentframe)
            motion_sum = sum(motionmatrix)/self.b.getBufferSize()
            if motion_sum > self.THRESHOLD:  
                print(motion_sum)
                hasmotion = True
                print("Mvt detected")
            else :
                hasmotion = False

        self.Nb_Frames += 1
        return hasmotion


class FrameBuffer(list):

    def __init__(self):
        #xxx
        list.__init__(self)
        self.BUFFER_SIZE = 15
        self.currentframe = None

    def motionImProc(self, cframe):
        #Apply different filter for image processing.
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(35,35))
        cframe = np.uint8(cframe)
        cframe = cv2.cvtColor(cframe, cv2.COLOR_RGB2GRAY)
        cframe = cv2.GaussianBlur(cframe, (61,61), 0)
        cframe = clahe.apply(cframe)
        return cframe

    def bufferReady(self):
        if len(self) == self.BUFFER_SIZE:
            return True

    def getBufferSize(self):
        return self.BUFFER_SIZE

    def appendFrame(self, cframe):
        self.currentframe = self.motionImProc(cframe)
        if (self.currentframe != None).any() and len(self) < self.BUFFER_SIZE:
            self.append(self.currentframe)
        if (self.currentframe != None).any() and len(self) >= self.BUFFER_SIZE:
            self.pop(0)
            self.append(self.currentframe)
        return self

class SaveMotion:
    #xxx
    def __init__(self, isboard):
        self.VIDEO_LENGTH = 10
        self.SavingState = False
        self.Record_During = 0
        self.IsBoard = isboard
        self.SavingState = False
        self.m = Motion()
        self.out = None
        self.IsFirst = True
        self.video_name = None

    def currentTime(self):
        #Return string with current timestamp
        return datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    
    def resetConditions(self):
        if self.Record_During == FPS * self.VIDEO_LENGTH:
            return True

    def reset(self):
        self.Record_During = 0
        self.SavingState = False 
        self.out.release()
        print("Reset Done")
        self.i=0
        self.IsBoard = 0
        os.chdir(f'{sync_path}')
        os.system(f'mkdir {self.video_name}')
        os.chdir(f'{self.video_name}')
        os.system(f'ffmpeg -hide_banner -loglevel error -i {video_output_path}/{self.video_name}.avi -r 1/1 %03d.jpg')
        os.system(f'rm {video_output_path}/{self.video_name}.avi')
        os.chdir(f"{predictor_path}")

    def crtBufferConditions(self, frame):
        #pdb.set_trace()
        if self.SavingState == False and self.m.hasMotion(frame) and self.IsBoard%2==0:
            return True
    
    def createBufferVideo(self):
        #Return a buffer VideoWriter object with timestamp as name. This is needed in Save_Video to properly save the video file locally. 
        self.SavingState = True
        timestamp=self.currentTime()
        self.video_name = f'yoli_data_{timestamp}'
        self.out = cv2.VideoWriter(f'video_output/{self.video_name}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (pred_width,pred_height))

    def savingConditions(self):
        if self.SavingState == True and self.Record_During<FPS*self.VIDEO_LENGTH:
            return True

    def saveVideo(self, image):
        #Saving the video properly
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.out.write(image)
        self.Record_During += 1
        self.IsBoard += 1
        
        if self.resetConditions():
            self.reset()

    def recSave(self, image):
        if self.crtBufferConditions(image):
            self.createBufferVideo()

        if self.savingConditions():
            self.saveVideo(image)

    def atBeginning(self, image):
        if self.IsFirst and self.IsBoard:
            self.createBufferVideo()
            self.saveVideo(image)
            self.IsFirst = not self.IsFirst

    def recording(self,frame):
        frame = np.uint8(frame)
        self.atBeginning(frame)
        self.recSave(frame)

class Inferencing:
    def __init__(self):
        #Model initialization and loading
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file("output/config.yaml")
        cfg.MODEL.DEVICE = "cpu"
        cfg.DATASETS.TEST = ("datasets/ImageNet2012/test", )
        cfg.MODEL.WEIGHTS = ("output/model_final.pth")
        self.model = DefaultPredictor(cfg)

    def prediction(self, frame): 
        t1=time.time()
        res = self.model(frame)
        label = res["pred_classes"]
        a = label.item()
        isboard = False 
        
        if a==0:
            #self.IsBG = 1
            text = "There's nothing"
            isboard = True
        elif a==1:
            text = "Correct"
        else :
            text = "Incorrect"
        t2=time.time()
        print(f"{1./(t2-t1)} fps")
        return (text, isboard)

class GstPipeline:
    #xxx
    def __init__(self, pipeline=None, saving_arg=None, prediction_arg=None, user_function=None, src_size=None, pipeline_out=None):
        #Gstreamer related variables
        self.running = False
        self.gstsample = None
        self.condition = threading.Condition()
        self.player = Gst.parse_launch(pipeline)
        self.saving_ON, self.prediction_ON = saving_arg, prediction_arg

        #Fetch different pads from pipeline for manipulation
        appsink = self.player.get_by_name("main-sink")
        appsink.connect("new-preroll", self.on_new_sample, True)
        appsink.connect("new_sample", self.on_new_sample, False)

        #Src pad in which to put the model output
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        self.run()

    def on_new_sample(self, sink, preroll):
        sample = sink.emit('pull-preroll' if preroll else "pull-sample")
        s = sample.get_caps().get_structure(0)
        with self.condition:
            self.gstsample = sample
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print(f"error: {err}, debug: {debug}")

    def run(self):
        self.running = True
        worker = threading.Thread(target=self.main_loop)
        worker.start()
        self.player.set_state(Gst.State.PLAYING)
        try:
            Gtk.main()
        except Exception as e:
            print(e)
            pass

    def label(self, model_output_label):
        #Edit label sink in the Gstreamer pipeline
        self.overlay = self.player.get_by_name('oover')
        self.overlay_text = model_output_label
        self.overlay.set_property("text",self.overlay_text)
        self.overlay.set_property("shaded-background","true")
        self.player.add(self.overlay)

    def main_loop(self):
        #Do inferencing or motion detection/recording given the flag
        print("Analyzing the stream ...")
        label_text, isboard = None, False
        s = SaveMotion(isboard)
        i = Inferencing()
        while True:
            with self.condition:
                while not self.gstsample and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstsample = self.gstsample
                self.gstsample = None
            gstbuffer = gstsample.get_buffer() #checked
            success, map_info = gstbuffer.map(Gst.MapFlags.READ) #checked
            if not success:
                raise RuntimeError("Could not map buffer data!")
            else:
                #get one frame
                sample_data = np.ndarray(
                        shape=(pred_height,pred_width,3),
                        dtype=np.uint8,
                        buffer=map_info.data) #checked

            if self.saving_ON:                
                s.recording(sample_data)

            if self.prediction_ON:
                pass
                label_text, isboard = i.prediction(sample_data)
                self.label(label_text)
            gstbuffer.unmap(map_info)

Gst.init(None)

parser = argparse.ArgumentParser(description='Detect objects from webcam images')
parser.add_argument('-s', '--save', type=int, default=0, help='Save to built data')
parser.add_argument('-p', '--prediction', type=int, default=0, help='Do inferencing')
args = parser.parse_args()    

GstPipeline(pipeline=pipeline,saving_arg = args.save, prediction_arg = args.prediction)
GObject.threads_init()
