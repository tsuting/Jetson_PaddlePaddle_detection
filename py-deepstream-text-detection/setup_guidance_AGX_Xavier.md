# Introduction

This document is the setup guidance of [Jetson AGX Xavier Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit). The author was using the old version (memory is 16 GB instead of 32 GB).

In this document, we will install the following things through NVIDIA SDK Manager 1.7.3.

* JetPack 4.6.1
* DeepStream SDK 6.0.1
* TensorRT 8.2
* Cuda 10.2
* Cudnn 8.2

Furthermore, we will build and install `Python bindings of deepStream SDK` on the device.

**This guidance might not be 100% correct and there might be other ways to do so (like Docker).**

## Prerequisite

* A machine with Ubuntu Linux x64 Version 18.04 or 16.04
* A USB keyboard and a mouse
* A display (either HDMI or DP)
* A network cable (or you can buy [a wireless card](https://forums.developer.nvidia.com/t/wi-fi-adapter-for-agx-xavier/127512))
* A Nvidia developer account

## Steps

1. Install JetPack by NVIDIA SDK Manager
    * Please follow [the official instruction#HOW TO INSTALL JETPACK](https://developer.nvidia.com/embedded/dlc/jetson-agx-xavier-developer-kit-user-guide) and [here](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html#install-with-sdkm-jetson)
    * Please remember to check DeepStream
    * The whole process took around 4 hours for my case which was definitely abnormal. Most of the errors were regarding network connection. Please check [the troubleshooting](#troubleshooting). If the error happens on your host computer, you can reinstall it later. However, if it happens on your device, then the installation will be stuck.

2. Check the deepstream installation
    * `deepstream-app --version-all`

3. (Optional) Run examples, the details for each example could be found [here](https://docs.nvidia.com/metropolis/deepstream/5.0/dev-guide/index.html#page/DeepStream_Development_Guide/deepstream_Sample_configs_and_Streams.html).
    * For example, if you have a USB webcam, `sudo deepstream-app -c /opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.tx`

4. (Optional) Change the power mode and the fan mode
    * Please refer [here](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0E0YO0HA) for more details.
    * Commands

        ```bash
        sudo nvpmodel -q  # Check the mode
        sudo nvpmodel deepstream -d cool # fan mode: cool
        sudo nvpmodel -m 0 # power mode: maximum 
        ```

5. (Optional) Setup VNC
   1. Why? As the author doesn't have enough space to put a display and a mouse for the device.
   2. Please follow [the instruction](https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup).
   3. Remember to **enable automatic login** on your device.
   4. If the device doesn't connect to any display, please remember to modify `/etc/X11/xorg.conf` as stated [here](https://forums.developer.nvidia.com/t/640x480-for-vnc-offer-more-choices/158713/4).

6. Build and install python bindings (not using docker, you could use docker)
   1. Please follow [the instruction](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/bindings/README.md).
   2. When you clone the repository, remember to use `git clone --recursive https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git` and put under `<DeepStream 6.0.1 ROOT>/sources/deepstream_python_apps`
   3. The command of building for the Jetson device is [this one](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings#323-example).

7. (Optional) Run python examples

   ```bash
   cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/deepstream-test1
   python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
   ```

## Troubleshooting

* Stuck in the installation of OpenCV on the device (reference from [here](https://github.com/SamKaiYang/Xavier_install#%E5%88%B7%E6%A9%9F%E5%B8%B8%E9%81%87%E5%88%B0%E7%9A%84%E5%95%8F%E9%A1%8C))
  * ssh into your device
  * Type the following commands

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt --fix-broken install
    ```

* Network errors e.g., connection failure
  * This happened many times during the installation. I found that even the error happened, it didn't affect the installation progress. However, if it got stuck here, please `ping 192.168.55.1` to see if the connection is okay. Furthermore, check your host computer's network using `ifconfig`. Please see [here](https://forums.developer.nvidia.com/t/unable-to-install-sdk-components-post-flashing-my-jetson-tx2-module/78380/21?page=2) for more details.

## Q&A

* How to check my camera information to find the available resolution and the frame rates?
  
    ```bash
    sudo apt-get install v4l-utils
    v4l2-ctl --list-devices # get the camera ID
    v4l2-ctl -d /dev/video0 --list-formats-ext # check the camera information
    ```

    OR `gst-device-monitor-1.0`

* How to check CPU/GPU usage?
  * [tegrastats](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html#)
  * [jetson_stats](https://github.com/rbonghi/jetson_stats)

* How to profile the application?
  * Nsight tool as stated [here](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#how-to-use-nvtx-for-profiling).
  * If you would like to use another host machine which is different from the one in the setup, please remember to check the version of your JetPack as stated from [here](https://forums.developer.nvidia.com/t/getting-target-is-not-supported-in-nvidia-nsight-systems-for-tx2/125704/8?u=yomizuya), *To profiling a Jetson device, the host tool must comes from the same SDKmanager. In other words, this indicates that only the Nsight System installed by the same JetPack version can profile Jetson device.*

## Useful resources

1. [Installation guidance and troubleshooting in Chinese](https://github.com/SamKaiYang/Xavier_install)
2. [Official DeepStream FAQ](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#frequently-asked-questions)
3. [Integrate TAO model with DeepStream SDK](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps)