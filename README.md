# Introduction 
This repo contains two tools that aim to work on a Raspberry Pi4 with a Logitech c930e plugged on. 
It is designed to work as a remote multi-class classifier.
The model is a Detectron2-based one, a direct implementation of https://github.com/tkianai/ImageClassification.detectron2 </br>
The predictor contains also a motion detection tool. Since, in my case the initial data was too small to train the Detectron2-based classifier, the motion detection has been designed to collect data remotely by using a Rasbperry Pi 4 and a webcam. This detector only intended to work until data is collected. 


### Options
The predictor contains 2 modes that can run at the same time. By default both modes are set to 0. 
-s flag is for the motion detection that allows video recording. The aim is to gather enough data for inferencing as preivously explained.
-p flag is for the inference of the classifier. Each frame of the webcam is extracted with GStreamer framework and goes into the inferencing box.

### Misc information
Motion detection is based on the subtraction of 2 grayscaled images after Clahe processing so with using OpenCV image processing tools. Thresholding can be adjust with MIN_CNT which adjusts the number of contours detected in the result of a subtraction.
If there is no movement during 15min, the pipeline stops.

Every recorded videos are sync with the Google server dedicated to the Yoli case. Sync is set up using systemd which runs (on boot) automatically a python script called rsync.py with a dedicated rsync command in it. The yoli_data.timer and yoli.data.service need to be moved to `/etc/systemd/system/`.
The timer can be run with the following command :
```
sudo systemctl start yoli_data.timer
```

To enable the timer on boot : 
```
sudo systemctl enable yoli_data.timer --now
```
As soon as the timer is launched, the .service runs automatically so you don't need to run it aferwards.

To check both .timer and .service :
```
sudo systemctl status yoli_data.service
sudo systemctl status yoli_data.timer 
```

# Usage

### For inferencing
```python predictor -s 1 -p 1```

### To view the video output from client side
```gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! autovideosink```

### For further training
```python predictor.py --model output/model_final.pth --config output/config.yaml```



# Beware
### This repo does not contain the datasets folder.
### The following steps/requirements are needed : 
- Raspbian OS aarch64 version (needed for torch and torchvision) : <br/>
    https://www.raspberrypi.org/forums/viewtopic.php?t=275370 <br/>

- Create a detectron2 environnement :
```
    python3 -m venv detectron2
 ```

(Inside the venv)
- PyTorch :
```
sudo apt-get update
sudo apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev
pip install --upgrade setuptools
pip install Cython
pip install gdown
gdown https://drive.google.com/uc?id=1PnKEuzg0JlqLKJBL6udZ-Mq44Buy0yaP
pip install torch-1.8.1a0+56b43f4-cp37-cp37m-linux_aarch64.whl
rm torch-1.8.1a0+56b43f4-cp37-cp37m-linux_aarch64.whl
```

- Torchvision :
```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev 
gdown https://drive.google.com/uc?id=1GSmkSWTaRMsRNaqR6PYeeH-vXRspHH3q
pip install torchvision-0.9.1a0+8fb5838-cp37-cp37m-linux_aarch64.whl
rm torchvision-0.9.1a0+8fb5838-cp37-cp37m-linux_aarch64.whl
```

- Detectron2 :
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
if there is a invalid command 'bdist_wheel' error ; uninstall an reinstall wheel 
pip uninstall wheel
pip install wheel
```

- GSTREAMER :
```
sudo apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-good python3-gst-1.0 python3-gi gir1.2-gtk-3.0 
sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
pip install pycairo
```

- OpenCV :
```
pip install opencv-python# detectron2_class
```
