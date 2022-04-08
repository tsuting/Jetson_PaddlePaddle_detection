# Introduction

Implement text detection with [the PaddlePaddle text detection model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/models_list_en.md#1-text-detection-model) and [object tracking](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) on deepStream using python bindings.

The code shows how to

* Use a USB webcam as an input
* Use RTSP/EGL/fakesink as an output
* Use different object trackers
* Use a customized parser
* Add performance measurement (fps)

Most of the code are copied from [`deepstream_python_apps/apps/`](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps). The code for PaddlePaddle decoder is from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/63cd23ab1c3f092666aad627a27dab1588aca83f/ppocr/postprocess/__init__.py) and [here](https://github.com/PaddlePaddle/PaddleOCR/blob/63cd23ab1c3f092666aad627a27dab1588aca83f/ppocr/postprocess/db_postprocess.py
).

## Prerequisite

* A USB webcam
* Follow [the guidance](setup_guidance_AGX_Xavier.md) to setup your device and environment if you're using Jetson AGX Xavier Developer Kit.
* Remember to add this folder under `deepstream_python_apps`.
* Place the onnx model under the same directory as the python files or please modify `onnx-file` in `ocr_pgie_config.txt`.
* If you would like to run DeepSORT object tracker, please follow the documentation to prepare the model.
  * `/opt/nvidia/deepstream/deepstream/sources/tracker_DeepSORT/README`
  * `dpkg -l | grep TensorRT` to check if `uff-converter-tf` and `graphsurgeon-tf` packages are installed.

## Parameters

```bash
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

## Prepare environments

`pip3 install -r ../requirements.txt`

## Run

`python3 deepstream_ocr.py -d /dev/video0`

## Troubleshooting

* When access RTSP, `Warning: g_object_get_is_valid_property: object class 'GstUDPSrc' has no property named 'pt'`
  * From [here](https://forums.developer.nvidia.com/t/warning-when-trying-to-view-the-rtsp-generated-by-deepstream/107184/5), *The print is harmless and please ignore it. We will check to fix it in future release.*

## Q&A

* Where does `net-scale-factor` and `offsets` come from? \
  These two are the parameters for preprocessing the frames. Please check [here](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer) for more details.
  
  In [the preprocessing step for PaddlePaddle text detection model](https://github.com/PaddlePaddle/PaddleOCR/blob/d9d6e23e3074c075d22da1e32956b4278502d92d/tools/infer/predict_det.py), the std is `[0.229, 0.224, 0.225]`, mean is `[0.485, 0.456, 0.406]`, and the scale is `1/255`. The normalization function is `(image*scale - mean)/std` from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/a67a6fa382cbf55aff12154d0fd3635c3a980b07/ppocr/data/imaug/operators.py#L115).
  
  In Gst-nvinfer, `y = net scale factor*(x-mean)` and `net-scale-factor` can be only one number.
  
  Therefore, mean (PaddlePaddle) is divided by the scale (PaddlePaddle) to get `mean` for Gst-nvinfer. Then the `net scale factor` would be the average of scale(PaddlePaddle)/std(PaddlePaddle).

* How to use a customized parser but not using Triton server? \
  The key is to set `network-type=100` and `output-tensor-meta=1` in `ocr_pgie_config.txt` as stated [here](https://forums.developer.nvidia.com/t/use-custom-behavior-cloning-neural-network-with-nvinfer/107827/4?u=yomizuya).

* How to get the output layers' names of onnx models? \
  https://netron.app/