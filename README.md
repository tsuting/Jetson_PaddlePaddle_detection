# Model Inference on Jetson Devices

## About this repo
This repo aims at showing examples of how to run inference on Nvidia Jetson devices, the code has been tested on both Jetson Nano (4GB version) and Jetson AGX Xavier Developer Kit (16 GB version).
In this repo, we show two different methods of doing inference using *python* and your *own model*:
- Using [Nvidia Deepstream framework](https://developer.nvidia.com/deepstream-sdk), see sample code in [this folder](py-deepstream-text-detection/)
- Using [TensorRT](https://developer.nvidia.com/tensorrt) directly with custom video processing code, see sample code in [this folder](py-tensorrt-text-detection/)

**Why do we want to have this repo?** </br>
When working with Nvidia python SDKs (especially Deepstream), we figured out there are many questions than answers.
Therefore, we decided to keep the record for future use purpose.



## About Device Setup
For setting up your Nvidia device, we've provided some [tips](py-deepstream-text-detection/setup_guidance_AGX_Xavier.md) for Jetson AGX Xavier Developer Kit. We *did not* provide guidance for Nano device because the setup process is similar and you will have a success if you just follow [the official setup documentation](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).


**Can I run the sample code using non-Jetson Device** </br>
Yes you can. However, you *must* make sure your device has an Nvidia GPU and satisfy the requirement of using TensorRT or Deepstream.

## About the Sample Application
This repository uses models from [PaddlePaddle OCR](https://github.com/PaddlePaddle/PaddleOCR), specifically, text detection only.
We provided a copy of the ONNX model used in this repo, see [here](models/ch_PP-OCRv2_det_infer.onnx).
If you have any questions related to the model, please directly contact [PaddlePaddle OCR](https://github.com/PaddlePaddle/PaddleOCR).


