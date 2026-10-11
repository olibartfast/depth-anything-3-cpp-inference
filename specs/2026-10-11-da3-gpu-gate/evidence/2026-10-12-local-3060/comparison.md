# 2026-10-12 local smoke — RTX 3060 Laptop 6GB, driver 610.43, CUDA UMD 13.3
# Model: /tmp/da3-small.onnx (DA3-Small, traced 504x504, opset 17)
# Image: /tmp/test504.png (sample_mid.png resized 1280x720 -> 504x504)

- C++ inference_app: Model input 504x504, Depth range [0.546271, 1.50737], outputs saved here.
- Python eager depth range: [0.54627097, 1.5073718] — matches C++ to 6 decimals.
- Parity eager-vs-ONNX (same preprocessed pixels): max-abs 5.662e-06, mean-abs 4.057e-07.
- Gate V-5 (mean <= 1e-3): PASS with ~2500x margin.
- Limitation: traced ONNX is fixed-shape 504x504 (RoPE tables frozen at trace time);
  720x1280 input fails in ORT Add broadcast (721 vs 1297). Dynamic-axes = declared, not honored.
