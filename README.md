# Intel-Raspberry-OpenGLES viewer
> Quick Description:  This solution has two modes, it should work from a recoring and from a decive

## Installation guide

#### Installing the OpenGL dependencies

1° First install dependencies

~~~sh
sudo apt-get update && sudo apt-get dist-upgrade
sudo apt install mesa-utils

sudo apt install -y \
    libopencv-dev \
    libeigen3-dev \
    libusb-1.0-0-dev \
    libglew-dev \
    freeglut3-dev \
    libglew2.1 \
    libopenal-dev \
    libglm-dev \
    libomp-dev

~~~
2° Check OpenGL version
~~~sh
glxinfo | grep "OpenGL version"
~~~

3° Export Display
~~~sh
export DISPLAY=:0
~~~

4° You can also follow this [video](https://www.youtube.com/watch?v=3dhDqLnWVb0&list=PLgpana-oqo-JO9hytkF3LC1HOX4v2OrGC) to install the dependencies and activate [VNC Server](https://www.realvnc.com/es/connect/download/vnc/)

#### Intel realsense Dependencies

Following this guide should be enough for [librealsense on raspberry Pi 4](https://github.com/mathklk/realsense_raspberry_pi4)

#### Extra dependencies
1. Cmake
2. OpenCV
3. Eigen