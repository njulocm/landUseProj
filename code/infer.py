import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import sys


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def infer_main(trt_path='../user_data/checkpoint/round2_b0_SmpUnetpp_9v1-0327/engine.trt'):
    MAX_BATCH = 1
    TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

    # Build an engine
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # 查看模型输入输出
    # for binding in engine:
    #     size = trt.volume(engine.get_binding_shape(binding)) * 1
    #     dims = engine.get_binding_shape(binding)
    #     dtype = trt.nptype(engine.get_binding_dtype(binding))
    #     print(size)
    #     print(dims)
    #     print(binding)
    #     print("input =", engine.binding_is_input(binding))
    #     print(dtype)
    #     print()

    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    # Do inference
    shape_of_output = (10, 256, 256)
    # Load data to the buffer
    inputs[0].host = np.random.uniform(-3, 3, [4, 256, 256]).astype(np.float32) # 随机生成测试数据
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
    trt_output = trt_outputs[0].reshape(shape_of_output)
    ret = np.argmax(trt_output, axis=0) + 1
    print('finish!')
    print(trt_output.shape)


if __name__ == '__main__':
    args = sys.argv
    trt_path = args[1]
    # ckpt_path = '../landUseProj/user_data/checkpoint/round2_b0_SmpUnetpp_9v1-0327/SmpUnetpp_best.trt'
    infer_main(trt_path=trt_path)
