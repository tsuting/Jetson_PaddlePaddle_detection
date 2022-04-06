#!/usr/bin/env python3

################################################################################
# Parts of the code are from deepstream-ssd-parser and deepstream-test1-usbcam
# The code performs customized parsing after the inference.
################################################################################

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import pyds
import argparse

# For the post-processing parsing function
import ctypes
import numpy as np
from paddle_decoder import td_PaddlePaddlePostProcess

# Performance
from common.FPS import GETFPS

# Global variables
UNTRACKED_OBJECT_ID = 0xffffffffffffffff

def pgie_src_pad_buffer_probe(pad, info, u_data):
    # 1. Get the data from the buffer
    # 2. Do post-processing e.g., extract bounding boxes
    # 3. Insert data to the meta data

    # Get the buffer of info argument.
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve the gst buffer from batch meta
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    # Retrieve the first frame
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:

            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone..

            # Retrieve the metadata for the first frame
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break  

        # Retrieve the frame meta user list and verify it is not None
        l_user = frame_meta.frame_user_meta_list  
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
            ):
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
       
            # Boxes in the tensor meta should be in network resolution which is
            # found in tensor_meta.network_info. Use this info to scale boxes to
            # the input frame resolution.
            
            # As we only have one output layer, the num_output_layers would be one.
            output_layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            
            # call the post-process function
            frame_object_list = nvds_infer_parse_custom_ocr(
                output_layer=output_layer
            )

            try:
                l_user = l_user.next
            except StopIteration:
                break

            for frame_object in frame_object_list:
                add_obj_meta_to_frame(frame_object, batch_meta, frame_meta)
            
        # Measure FPS, this inlcudes the time drawing the boxes
        fps_stream.get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def nvds_infer_parse_custom_ocr(output_layer):
    
    # Reshape the output and convert to numpy
    # If you use pyds.get_detections(num_detection_layer.buffer, index), it will only extract one index every time. 
    # Therefore, use the following way. 
    # Ref: https://forums.developer.nvidia.com/t/urgent-how-to-convert-deepstream-tensor-to-numpy/128601/13
    ptr = ctypes.cast(pyds.get_ptr(output_layer.buffer), ctypes.POINTER(ctypes.c_float)) 
    data = np.ctypeslib.as_array(ptr, shape=(1088*1440,))
    data = data.reshape(1, 1, 1, 1088, 1440)
    # post-process
    paddle = td_PaddlePaddlePostProcess()
    # result is a np.array which contains boundind boxes in one frame. 
    result = paddle(results=data, shape_list=[[1088, 1440, 1, 1]])

    object_list = []
    # loop through each bounding box and add to object_list
    # TODO: Can't we direclty add to nvds_add_obj_meta_to_frame in the next function?
    for i in range(result.shape[0]):
        # https://docs.nvidia.com/metropolis/deepstream/5.0.1/sdk-api/Gst_Infer/NvDsInferObjectDetectionInfo.html
        res = pyds.NvDsInferObjectDetectionInfo()
        res.detectionConfidence = 1
        res.classId = 0
        res.left = min(result[i][:, 0])
        res.top = min(result[i][:, 1])
        res.width = abs(max(result[i][:, 0]) - min(result[i][:, 0]))
        res.height = abs(max(result[i][:, 1]) - min(result[i][:, 1]))
        object_list.append(res)

    return object_list

def add_obj_meta_to_frame(frame_object, batch_meta, frame_meta):
    """ Inserts an object into the metadata """
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
    rect_params.left = frame_object.left
    rect_params.top = frame_object.top
    rect_params.width = frame_object.width
    rect_params.height = frame_object.height

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # Set object info including class, detection confidence, etc.
    obj_meta.confidence = frame_object.detectionConfidence
    obj_meta.class_id = frame_object.classId

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    obj_meta.object_id = UNTRACKED_OBJECT_ID

    # Set the object classification label.
    obj_meta.obj_label = "text"

    # Set display text for the object.
    txt_params = obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)
    
    txt_params.x_offset = int(rect_params.left)
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    txt_params.display_text = (
        f"{obj_meta.obj_label}"
    )
    
    # Font , font-color and font-size
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    # set(red, green, blue, alpha); set to White
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Text background color
    txt_params.set_bg_clr = 1
    # set(red, green, blue, alpha); set to Black
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # Required if you're using a customized parsing function.
    # https://forums.developer.nvidia.com/t/deepstream-6-with-python-binding-how-to-apply-tracking-on-custom-inferserver-results-in-deepstream-ssd-parser-sample/203878/6
    # bInferDone – int, Boolean indicating whether inference is performed on given frame.
    frame_meta.bInferDone = 1
    
    # Insert the object into current frame meta
    # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = f"Frame Number={frame_number} Number of Objects={num_rects}"

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print (pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # # An example of saving a raw frame
        # from pathlib import Path
        # import cv2
        # n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        # # convert python array into numpy array format in the copy mode.
        # frame_copy = np.array(n_frame, copy=True, order='C')
        # # convert the array into cv2 default color format
        # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
        # # create a folder
        # Path("output").mkdir(exist_ok=True)
        # cv2.imwrite(f"output/{frame_meta.frame_num}.jpg", frame_copy)

        # # If you would like to save a frame including bounding boxes,
        # # please check deepstream-imagedata-multistream-redaction
        # # It looks like there is no way to save the same directly. 
        # # You need to draw boudning boxes by yourselves. (Or maybe try to catch frames from rtsp?)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	


def main():
    # Initialize for the FPS measurement 
    global fps_stream
    fps_stream = GETFPS(0)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    # Source element for reading from the camera
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")
    # Remember to change here if your device is not /dev/video0
    # You can use this command to check $ v4l2-ctl --list-devices 
    source.set_property('device', device_path)
    pipeline.add(source)

    # Use v4l2-ctl -d /dev/video0 --list-formats-ext or gst-device-monitor-1.0
    # You don't need to speficy width and height. It's just an example.
    # You can check the default values by frameMeta
    # https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html#pyds.NvDsFrameMeta
    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")
    caps_v4l2src.set_property('caps', Gst.Caps.from_string(f"video/x-raw, framerate=30/1"))
    pipeline.add(caps_v4l2src)
    source.link(caps_v4l2src)

    print("Creating Video Converter \n")
    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)

    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")
    pipeline.add(vidconvsrc)
    caps_v4l2src.link(vidconvsrc)

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
    pipeline.add(nvvidconvsrc)
    vidconvsrc.link(nvvidconvsrc)

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    pipeline.add(caps_vidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    streammux.set_property('width', 1440)
    streammux.set_property('height', 1088)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pipeline.add(streammux)
    # link
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    pgie.set_property('config-file-path', "ocr_pgie_config.txt")
    pipeline.add(pgie)
    streammux.link(pgie)

   # Add a probe on the primary-infer source pad to get inference output tensors
    pgiesrcpad = pgie.get_static_pad("src")
    if not pgiesrcpad:
        sys.stderr.write(" Unable to get src pad of primary infer \n")
    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Add an object tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    
    # each frame (not bboxes) will be scaled to the following size.
    tracker.set_property('tracker-width', 1440)
    tracker.set_property('tracker-height', 1088)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    if object_tracker == "NvDCF":
        tracker.set_property('ll-config-file', "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml")
    elif object_tracker == "DeepSORT": 
        # Remember to follow the instruction to convert and place the model
        # /opt/nvidia/deepstream/deepstream/sources/tracker_DeepSORT/README
        tracker.set_property('ll-config-file', "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_DeepSORT.yml")
    elif object_tracker == "IOU":
        tracker.set_property('ll-config-file', "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml")

    pipeline.add(tracker)
    pgie.link(tracker)

    if output_type == "egl" or output_type == "rtsp":
        # Draw bboxes
        # Use convertor to convert from NV12 to RGBA as required by nvosd
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")
        pipeline.add(nvvidconv)
        tracker.link(nvvidconv)

        # Create OSD to draw on the converted RGBA buffer
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")
        pipeline.add(nvosd)
        nvvidconv.link(nvosd)

        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

        if output_type == "egl":
            # Finally render the osd output        
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")

            # Set sync = false to avoid late frame drops at the display-sink
            sink.set_property('sync', False)
            pipeline.add(sink)
            
            if is_aarch64():
                # TODO: Why the following order?
                # pipeline: nvosd -> sink -> transform
                # link: nvosd -> tranform -> sink
                transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
                pipeline.add(transform)
                nvosd.link(transform)
                transform.link(sink)
            else:
                nvosd.link(sink)
                
        elif output_type == "rtsp":
            # The following code is derived from deepstream-test1-rtsp-out
            # TODO: Investigate on why we need to convert again?
            nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
            if not nvvidconv_postosd:
                sys.stderr.write(" Unable to create nvvidconv_postosd \n")
            pipeline.add(nvvidconv_postosd)
            nvosd.link(nvvidconv_postosd)

            # Create a caps filter
            caps_rtsp = Gst.ElementFactory.make("capsfilter", "filter")
            caps_rtsp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            pipeline.add(caps_rtsp)
            nvvidconv_postosd.link(caps_rtsp)
            
            # Make the encoder
            if rtsp_codec == "H264":
                encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
                print("Creating H264 Encoder")
            elif rtsp_codec == "H265":
                encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
                print("Creating H265 Encoder")
            if not encoder:
                sys.stderr.write(" Unable to create encoder")
            encoder.set_property('bitrate', rtsp_bitrate)
            if is_aarch64():
                encoder.set_property('preset-level', 1)
                encoder.set_property('insert-sps-pps', 1)
                encoder.set_property('bufapi-version', 1)
            pipeline.add(encoder)
            caps_rtsp.link(encoder)
            
            # Make the payload-encode video into RTP packets
            if rtsp_codec == "H264":
                rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
                print("Creating H264 rtppay")
            elif rtsp_codec == "H265":
                rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
                print("Creating H265 rtppay")
            if not rtppay:
                sys.stderr.write(" Unable to create rtppay")
            pipeline.add(rtppay)
            encoder.link(rtppay)

            # Make the UDP sink
            updsink_port_num = 5400
            sink = Gst.ElementFactory.make("udpsink", "udpsink")
            if not sink:
                sys.stderr.write(" Unable to create udpsink")
            sink.set_property('host', '224.224.255.255')
            sink.set_property('port', updsink_port_num)
            sink.set_property('async', False)
            sink.set_property('sync', 1)
            pipeline.add(sink)
            rtppay.link(sink)

            # Start streaming
            rtsp_port_num = 8554
            
            server = GstRtspServer.RTSPServer.new()
            server.props.service = "%d" % rtsp_port_num
            server.attach(None)
            
            factory = GstRtspServer.RTSPMediaFactory.new()
            factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, rtsp_codec))
            factory.set_shared(True)
            server.get_mount_points().add_factory("/ds-test", factory)
            
            print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    elif output_type == "none":
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        pipeline.add(sink)
        tracker.link(sink)    

        
    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='OCR Application Help')
				  
    parser.add_argument("-d", "--device_path", type=str, required=True,
                  help='Path to your camera device e.g. /dev/video0. Please use "v4l2-ctl --list-devices" to check.')	
    parser.add_argument("-c", "--codec", default="H264",
                  help="(only if the output is RTSP) RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="(only if the output is RTSP) Set the encoding bitrate, default=4000000", type=int)
    parser.add_argument("-o", "--output", default="rtsp",
                  help="The output type. If the output is none, nvosd will be skipped, default=rtsp", choices=['rtsp','egl', 'none'])    
    parser.add_argument("-t", "--object_tracker", default="NvDCF",
                  help="The approach to do objec tracking, default=NvDCF", choices=['NvDCF','DeepSORT', 'IOU'])   

    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Assign variables
    global device_path 
    global rtsp_codec
    global rtsp_bitrate
    global output_type
    global object_tracker
    device_path = args.device_path
    rtsp_codec = args.codec
    rtsp_bitrate = args.bitrate
    output_type = args.output
    object_tracker = args.object_tracker


if __name__ == '__main__':
    parse_args()
    sys.exit(main())

