import os
import subprocess
import sys

import torch

# Add current directory to path so models can be imported natively
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.unet_model import UNet


def export_to_onnx(pytorch_path, onnx_path):
    print(f"Loading PyTorch model from {pytorch_path}...")
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    state_dict = torch.load(pytorch_path, map_location="cpu")
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Matching UNET_INPUT_SIZE in config (256, 256)
    dummy_input = torch.randn(1, 3, 256, 256)

    print(f"Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # 11 or 14 is typical for TensorRT
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("ONNX export complete.")


def convert_onnx_to_trt(onnx_path, engine_path):
    print(f"Converting {onnx_path} to TensorRT {engine_path} using trtexec...")

    # Check default Jetson TensorRT bin path
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path):  # fallback to system PATH
        trtexec_path = "trtexec"

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",  # Enable FP16 optimization
    ]
    try:
        subprocess.run(cmd, check=True)
        print("TensorRT engine generation complete!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate TensorRT engine. Check if 'trtexec' is installed. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"trtexec command not found at {trtexec_path}. Are you inside the Jetson TensorRT container?")
        sys.exit(1)


if __name__ == "__main__":
    pth_model = "unet_msfd_model_best.pth"
    onnx_model = "unet_msfd.onnx"
    engine_model = "unet_msfd.engine"

    export_to_onnx(pth_model, onnx_model)
    convert_onnx_to_trt(onnx_model, engine_model)
