from typing import Dict, List
from pycuda import autoinit
import pycuda.driver as cuda
from pathlib import Path
import tensorrt as trt
import numpy as np

DEFAULT_DB_PROFILE = [{"x": [(1, 3, 1088, 1440)]}]
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()


class TRTUtils:
    def __init__(self, config: Dict) -> None:
        """
        init utils for tensorrt engine building
        config is in the following format
        {
            "fp16":True,
            "original_model_path":  "./tmp_engine/ch_pp_inf_dynamic.onnx",
            "engine_path":"./tmp_engine/ch_pp_inf_dynamic_fp16.engine",
            "dynamic_shape": True,
            "profile_config":[ # the shape is min_shape, opt_shape and max_shape order
                {"x" : [(1, 3, 960, 960), (1, 3, 1280, 1280), (1, 3, 1536,1536) ] }
            ],
        }

        :param config: configurations for building tensorrt
        :type config: Dict
        """
        self._trt_engine = None
        self._fp16 = config.get("fp16", False)
        self._original_model_path = config.get("original_model_path", "")
        self._engine_path = config.get("engine_path", "./inference.engine")
        self._dynamic_shape = config.get("dynamic_shape", False)
        self._profile_config = config.get("profile_config", DEFAULT_DB_PROFILE)
        self._output_shape = config.get("output_shape", None)
        self._trt_context = None

    def _build_engine_onnx(self) -> trt.tensorrt.ICudaEngine:
        """
        build tensorrt engine based on the configuration  for onnx model

        :return: tensorrt engine
        :rtype: trt.tensorrt.ICudaEngine
        """
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            if self._fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            builder.max_batch_size = 1
            # Parse model file
            if not Path(self._original_model_path).exists():
                print(f"ONNX file {self._original_model_path} is not found.")
                exit(1)
            print("Loading ONNX file from path {}...".format(self._original_model_path))
            with open(self._original_model_path, "rb") as onnx_buffer:
                if not parser.parse(onnx_buffer.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            if not self._dynamic_shape:
                profile_config = self._profile_config[0]
                for _, shape_list in profile_config.items():
                    network.get_input(0).shape = shape_list[0]
            else:
                for profile_config in self._profile_config:
                    profile = builder.create_optimization_profile()
                    for layer_name, shape_list in profile_config.items():
                        min_shape, opt_shape, max_shape = shape_list
                        profile.set_shape(layer_name, min_shape, opt_shape, max_shape)
                    config.add_optimization_profile(profile)
            print(
                "Building an engine from file {}; this may take a while...".format(
                    self._original_model_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(self._engine_path, "wb") as f:
                f.write(plan)
            return engine

    def get_engine(self):
        """
        get inference engine
        """
        if Path(self._engine_path).exists():
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self._engine_path))
            with open(self._engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())
        else:
            if Path(self._original_model_path).suffix == ".onnx":
                self._trt_engine = self._build_engine_onnx()
        self._trt_context = self._trt_engine.create_execution_context()

    def inference_single(
        self, input_data: np.ndarray, profile_idx: int = 0, profiling: bool = False
    ) -> List:
        """
        perform inference on the given input data
        note the input data is not a list

        :param input_data: input data to be predict on (after preprocessing)
        :type input_data: np.ndarray

        :param profile_idx: profile index, when dynamic shape is used, defaults to 0
        :type profile_idx: int, optional
        :param profiling: whether to run profiler, defauts to false
        :type profiling: bool, optional
        :return: list of prediction output
        :rtype: List
        """
        input_shape = input_data.shape

        inputs = []
        outputs = []
        bindings = []
        input_binding_idx = 0
        if profiling:
            profiler = trt.tensorrt.Profiler()
            self._trt_context.profiler = profiler

        # The following logic is hard-coding for paddlepaddle detection model and the recognition model
        # There is only one input and one output always.

        # Get input memory allocation, attention: we're using engine here
        binding_shape = list(
            self._trt_engine.get_binding_shape(index=input_binding_idx)
        )

        if self._dynamic_shape:
            binding_shape[-2:] = input_shape[-2:]
        host_mem, device_mem = self.calculate_memory_from_binding(
            binding_shape=binding_shape,
            binding_type=self._trt_engine.get_binding_dtype(index=input_binding_idx),
            batch_size=self._trt_engine.max_batch_size,
        )
        inputs.append(HostDeviceMem(host_mem, device_mem))
        bindings.append(int(device_mem))

        stream = cuda.Stream()
        if self._dynamic_shape:
            self._trt_context.set_optimization_profile_async(profile_idx, stream.handle)
        self._trt_context.set_binding_shape(input_binding_idx, input_shape)
        inputs[0].host = np.ascontiguousarray(input_data)
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        # Get output memory allocation, attention: we're using context + engine here
        output_binding_idx = len(self._trt_engine) - 1
        binding_shape = self._trt_context.get_binding_shape(binding=output_binding_idx)
        host_mem, device_mem = self.calculate_memory_from_binding(
            binding_shape=list(binding_shape),
            binding_type=self._trt_engine.get_binding_dtype(index=output_binding_idx),
            batch_size=self._trt_engine.max_batch_size,
        )
        outputs.append(HostDeviceMem(host_mem, device_mem))
        bindings.append(int(device_mem))

        self._trt_context.execute_async_v2(
            bindings=bindings, stream_handle=stream.handle
        )
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        output_buffer = [out.host for out in outputs]
        return output_buffer

    def calculate_memory_from_binding(self, binding_shape, binding_type, batch_size):
        """
        calculate memory allocation

        :param binding_shape: the binding shape
        :type input_data: List
        :param binding_type: the type of the binding
        :type binding_type: tensorrt.tensorrt.DataType
        :param binding_shape: the binding shape
        :type batch_size: int
        :param batch_size: the batch size

        :return: host memory and device memory
        :rtype: numpy.ndarray, numpy.ndarray
        """
        size = abs(trt.volume(tuple(binding_shape))) * batch_size
        dtype = trt.nptype(binding_type)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        return host_mem, device_mem


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
