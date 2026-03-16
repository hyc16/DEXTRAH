import torch
from ultralytics.models.sam.predict import SAM2Predictor


def run(device_str="cuda:0", model_path="sam2.1_l.pt"):
    assert torch.cuda.is_available(), "CUDA not available"
    dev_idx = int(device_str.split(":")[1])
    torch.cuda.set_device(dev_idx)

    predictor = SAM2Predictor(overrides={
        "model": model_path,
        "device": device_str,
        "imgsz": 1024,
        "mode": "predict",
        "task": "segment",
    })

    predictor.setup_model(model=None, verbose=False)

    print("=" * 80)
    print(f"[move] requested={device_str}")
    print(f"[move] before move model_device={next(predictor.model.parameters()).device}")
    print(f"[move] predictor.args.device={getattr(predictor.args, 'device', None)}")
    print(f"[move] predictor.device(before)={getattr(predictor, 'device', None)}")

    predictor.model = predictor.model.to(device_str)
    predictor.device = torch.device(device_str)
    predictor.args.device = device_str

    predictor.args.imgsz = 1024
    predictor.imgsz = (1024, 1024)
    predictor.setup_source(None)

    model_device = next(predictor.model.parameters()).device
    print(f"[move] after move model_device={model_device}")
    print(f"[move] predictor.device(after)={getattr(predictor, 'device', None)}")

    imgs = torch.randn(2, 3, 1024, 1024, device=device_str, dtype=torch.float32)
    print(f"[move] imgs.device={imgs.device}")

    with torch.inference_mode():
        _ = predictor.model.forward_image(imgs)

    print("[move] forward ok")


if __name__ == "__main__":
    for i in range(min(torch.cuda.device_count(), 8)):
        dev = f"cuda:{i}"
        try:
            run(dev)
        except Exception as e:
            print(f"[move] {dev} failed: {repr(e)}")