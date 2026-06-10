import numpy as np


def load_model(model_path: str):
    """Loads ONNX model with best available execution provider."""
    import onnxruntime as ort

    providers_priority = [
        'ROCMExecutionProvider',
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
    available = ort.get_available_providers()
    providers  = [p for p in providers_priority if p in available] or ['CPUExecutionProvider']

    print(f"Available ONNX providers: {available}")
    print(f"Using: {providers[0]}")

    session = ort.InferenceSession(model_path, providers=providers)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Input:  {inp.name!r} | shape: {inp.shape} | type: {inp.type}")
    print(f"Output: {out.name!r} | shape: {out.shape} | type: {out.type}")
    return session


def get_model_info(session) -> dict:
    """
    Detects model type, input name, input size, and scale factor.

    Returns dict with:
        input_name  : str
        output_name : str
        fixed_size  : (W, H) or None — None means dynamic input
        scale       : int (2, 3, 4...)
        is_zoo      : bool — True if input dimensions are fixed integers
    """
    inp   = session.get_inputs()[0]
    out   = session.get_outputs()[0]
    name  = inp.name
    shape = inp.shape

    h_dim = shape[2] if len(shape) > 2 else None
    w_dim = shape[3] if len(shape) > 3 else None
    is_fixed = isinstance(h_dim, int) and isinstance(w_dim, int)

    if is_fixed:
        fixed_h, fixed_w = int(h_dim), int(w_dim)
        dummy = np.zeros((1, 1, fixed_h, fixed_w), dtype=np.float32)
        dummy_out = session.run(None, {name: dummy})[0]
        scale = dummy_out.shape[2] // fixed_h
        return {
            'input_name':  name,
            'output_name': out.name,
            'fixed_size':  (fixed_w, fixed_h),
            'scale':       scale,
            'is_zoo':      True,
        }
    else:
        dummy = np.zeros((1, 1, 64, 64), dtype=np.float32)
        dummy_out = session.run(None, {name: dummy})[0]
        scale = dummy_out.shape[2] // 64
        return {
            'input_name':  name,
            'output_name': out.name,
            'fixed_size':  None,
            'scale':       scale,
            'is_zoo':      False,
        }
