# Introduction 
OCR with the paddlepaddle detection model and object tracking on deepStream. Input is from an usb camera. 

The code shows how  to
* Use usb camera as an input
* Use RTSP or EGL as an output
* Use different object trackers
* Use a customized parser
* Add performance measurement (fps)

# Prerequisite

  * Remember to add this folder under `deepstream_python_apps`
  * Place the onnx model under the same layer as the python files
  * If you would like to run DeepSORT object tracker, please follow the documentation to prepare the model.
    * /opt/nvidia/deepstream/deepstream/sources/tracker_DeepSORT/README

# Parameters

```python 
usage: deepstream_ocr.py [-h] -d DEVICE_PATH [-c {H264,H265}] [-b BITRATE]
                         [-o {rtsp,egl,none}] [-t {NvDCF,DeepSORT,IOU}]

OCR Application Help

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_PATH, --device_path DEVICE_PATH
                        Path to your camera device e.g. /dev/video0. Please
                        use "v4l2-ctl --list-devices" to check.
  -c {H264,H265}, --codec {H264,H265}
                        (only if the output is RTSP) RTSP Streaming Codec
                        H264/H265 , default=H264
  -b BITRATE, --bitrate BITRATE
                        (only if the output is RTSP) Set the encoding bitrate,
                        default=4000000
  -o {rtsp,egl,none}, --output {rtsp,egl,none}
                        The output type. If the output is none, nvosd will be
                        skipped, default=rtsp
  -t {NvDCF,DeepSORT,IOU}, --object_tracker {NvDCF,DeepSORT,IOU}
                        The approach to do objec tracking, default=NvDCF
```

# Run
`python3 deepstream_ocr.py -d /dev/video0`

# Troubleshooting

* When access RTSP, `Warning: g_object_get_is_valid_property: object class 'GstUDPSrc' has no property named 'pt'`
  * From [here](https://forums.developer.nvidia.com/t/warning-when-trying-to-view-the-rtsp-generated-by-deepstream/107184/5), *The print is harmless and please ignore it. We will check to fix it in future release.*