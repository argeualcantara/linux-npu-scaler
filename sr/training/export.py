import torch


def export_onnx(model, output_path: str, scale: int = 2):
    """
    Exports the trained model to ONNX with dynamic spatial dimensions.
    Dynamic axes allow the model to accept any resolution at inference time.

    Training pair:  720p LR  → 1440p HR  (2x, native 1440p captures)
    Inference:      540p  → 1080p  (Ally X), 720p → 1440p (PC)
    """
    model.eval()
    model_cpu = model.to('cpu')

    dummy = torch.randn(1, 1, 720, 1280)

    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        opset_version=17,
        input_names=['lr_frame'],
        output_names=['sr_frame'],
        dynamic_axes={
            'lr_frame': {0: 'batch', 2: 'height', 3: 'width'},
            'sr_frame': {0: 'batch', 2: 'height', 3: 'width'},
        },
        export_params=True,
        do_constant_folding=True,
    )
    print(f"\nModel exported to: {output_path}")

    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("ONNX validation: OK")
    except ImportError:
        print("(install 'onnx' to validate: pip install onnx)")
    except Exception as e:
        print(f"ONNX validation warning: {e}")
