import torch
import onnx
from onnxsim import simplify
import sys
import os


def ckpt2onnx(ckpt_path, onnx_path):
    print("start converting ckpt to onnx...")
    model = torch.load(ckpt_path).cuda()
    model.model.encoder.set_swish(False)
    model.eval()
    data = torch.rand(1, 4, 256, 256).cuda()
    input_names = ["input"]
    output_names = ["output"]
    out = torch.onnx.export(model, data, onnx_path, verbose=True, opset_version=11, input_names=input_names,
                            output_names=output_names)
    print(f"successfully convert {ckpt_path} to {onnx_path}")


def onnx2onnxsim(onnx_path, onnxsim_path):
    print("start converting onnx to onnxsim...")
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnxsim_path)
    print(f"successfully convert {onnx_path} to {onnxsim_path}")


def torch2trt(ckpt_path, FLOAT=16):
    # 1.convert ckpt to onnx
    onnx_path = ckpt_path.split('.pth')[0] + '.onnx'
    ckpt2onnx(ckpt_path, onnx_path)
    # 2. convert onnx to onnxsim
    onnxsim_path = onnx_path.split('.onnx')[0] + '-sim.onnx'
    onnx2onnxsim(onnx_path, onnxsim_path)

    # 3. convert onnxsim to trt
    MAX_BATCH = 1
    trt_path = ckpt_path.split('.pth')[0] + '.trt'
    # os.system('export PATH=$PATH:/workspace/onnx-tensorrt/build')
    os.system(f"/workspace/onnx-tensorrt/build/onnx2trt {onnxsim_path} -o {trt_path} -b {MAX_BATCH} -d {FLOAT}")


if __name__ == '__main__':
    args = sys.argv
    ckpt_path = args[1]
    # ckpt_path = '../landUseProj/user_data/checkpoint/round2_b0_SmpUnetpp_9v1-0327/SmpUnetpp_best.pth'
    torch2trt(ckpt_path=ckpt_path)
