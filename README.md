# Introduction 
This is an implementation of the detectron2 applied to multi-class classifier. 
The training can be done on both CPU/GPU with an appropriate machine and command.
It also has been tested on the RPi4 with Logitech c930e.
The backbone has been inspired from https://github.com/tkianai/ImageClassification.detectron2
Model has been trained for 3 classes for this Yoli case: Correct, Incorrect and Nothing (means no board)

### Options
There are 2 modes and they can run at the same time. Both flags to diable or enable them is 0 or 1. By default, both are set to 0.
-s flag is for saving video. The aim is to gather enough data for inferencing.
-p flag is for the inference. Each frame of the webcam is extracted with Gstreamer and goes into the inferencing box.

### Motion detection
Motion detection is based on the subtraction of 2 grayscaled images after Clahe processing so with using OpenCV image processing tools. Thresholding can be adjust with MIN_CNT which adjusts the number of contours detected in the result of a subtraction.
If there is no movement during 15min, the pipeline stops.

### Data sync
Every recorded videos are sync with the Google server dedicated to the Yoli case. Sync is set up using systemd which runs (on boot) automatically a python script called rsync.py with a dedicated rsync command in it. The yoli_data.timer and yoli.data.service need to be moved to /etc/systemd/system/.<br/>
The timer can be run with the following command : <br/>
sudo systemctl start yoli_data.timer <br/>
To enable the timer on boot : <br/>
sudo systemctl enable yoli_data.timer --now <br/>
As soon as the timer is launched, the .service runs automatically so you don't need to run it aferwards. <br/>
To check both .timer and .service : <br/>
sudo systemctl status yoli_data.service <br/>
sudo systemctl status yoli_data.timer 

# Usage

### For inferencing
python predictor -s 1 -p 1

### To view the video output from client side
gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! autovideosink

### For further training
python predictor.py --model output/model_final.pth --config output/config.yaml



# Beware
### This repo does not contain the datasets folder.
### The following steps/requirements are needed : 
- Raspbian OS aarch64 version (needed for torch and torchvision) : <br/>
    https://www.raspberrypi.org/forums/viewtopic.php?t=275370 <br/>

- Create a detectron2 environnement : <br/>
    python3 -m venv detectron2 <br/>

(Inside the venv)
- PyTorch : <br/>
1. sudo apt-get update <br/>
2. sudo apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev <br/>
3. pip install --upgrade setuptools <br/>
4. pip install Cython <br/>
5. pip install gdown <br/>
6. gdown https://drive.google.com/uc?id=1PnKEuzg0JlqLKJBL6udZ-Mq44Buy0yaP <br/>
7. pip install torch-1.8.1a0+56b43f4-cp37-cp37m-linux_aarch64.whl <br/>
8. rm torch-1.8.1a0+56b43f4-cp37-cp37m-linux_aarch64.whl <br/>


- Torchvision : <br/>
1. sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev <br/>
2. sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev <br/>
3. gdown https://drive.google.com/uc?id=1GSmkSWTaRMsRNaqR6PYeeH-vXRspHH3q <br/>
4. pip install torchvision-0.9.1a0+8fb5838-cp37-cp37m-linux_aarch64.whl <br/>
5. rm torchvision-0.9.1a0+8fb5838-cp37-cp37m-linux_aarch64.whl <br/>

- Detectron2 : <br/>
1. python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' <br/>
2. if there is a invalid command 'bdist_wheel' error ; uninstall an reinstall wheel : <br/>
3. pip uninstall wheel <br/>
4. pip install wheel <br/>

- GSTREAMER : <br/>
1. sudo apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-good python3-gst-1.0 python3-gi gir1.2-gtk-3.0 <br/>
2. sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 <br/>
3. pip install pycairo <br/>

- OpenCV : <br/>
1.  pip install opencv-python# detectron2_class
