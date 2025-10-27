"""
ONNX Export Script
==================
Export trained TCN model to ONNX format for deployment.
"""

import argparse
import torch
import onnx
from pathlib import Path

from models.tcn import TemporalCNN, TCNConfig
from training.trainer import IDSTrainer


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_dim: int = 25,
    window_size: int = 128,
    opset_version: int = 14
):
    """Export PyTorch model to ONNX"""

    print("="*70)
    print("ONNX Export")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input dim: {input_dim}")
    print(f"Window size: {window_size}")
    print(f"ONNX opset: {opset_version}")
    print("="*70)

    # Load model
    print("\n📦 Loading model...")
    try:
        model = IDSTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu'
        )
        print("✅ Loaded as IDSTrainer")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Loading manually...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config
        if 'hyper_parameters' in checkpoint:
            input_dim = checkpoint['hyper_parameters'].get('input_dim', input_dim)

        # Create model with config
        config = TCNConfig(
            input_dim=input_dim,
            num_channels=[256, 256, 512, 512],
            kernel_size=3,
            num_classes=2,
            dropout=0.1
        )
        model_net = TemporalCNN(config)

        # Load weights
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_k] = v
        model_net.load_state_dict(new_state_dict, strict=False)

        model = IDSTrainer(model_net, learning_rate=1e-3)
        print("✅ Loaded manually")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, window_size, input_dim)
    print(f"\n📊 Dummy input shape: {dummy_input.shape}")

    # Export to ONNX
    print(f"\n🔄 Exporting to ONNX...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ ONNX export complete!")

    # Verify ONNX model
    print(f"\n🔍 Verifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model is valid!")

    # Model info
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n📊 Model Info:")
    print(f"  File: {output_path.name}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Path: {output_path.absolute()}")

    # Test inference
    print(f"\n🧪 Testing ONNX inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    outputs = session.run(None, inputs)

    print(f"✅ ONNX inference successful!")
    print(f"  Output shape: {outputs[0].shape}")

    print("\n" + "="*70)
    print("✅ Export Complete!")
    print("="*70)
    print(f"\nUsage:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path.name}')")
    print(f"  outputs = session.run(None, {{'input': features}})")


def main():
    parser = argparse.ArgumentParser(description='Export TCN to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file')
    parser.add_argument('--input-dim', type=int, default=25, help='Input feature dimension')
    parser.add_argument('--window-size', type=int, default=128, help='Window size')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version')

    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_dim=args.input_dim,
        window_size=args.window_size,
        opset_version=args.opset
    )


if __name__ == '__main__':
    main()
