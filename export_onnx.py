"""Export weights/efficientnet_b0.pth to weights/efficientnet_b0.onnx (needs torch + onnx)."""
import numpy as np
import torch

from models import CalorieNet


class _Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out["classification"], out["mass"].reshape(-1, 1)


def main(out_path: str = "weights/efficientnet_b0.onnx"):
    model, _ = CalorieNet.from_checkpoint("weights/efficientnet_b0.pth", device="cpu")
    model.eval()
    # CalorieNet overrides train()/eval() with no-arg versions; restore the stock nn.Module one for export.
    CalorieNet.train = torch.nn.Module.train
    CalorieNet.eval = torch.nn.Module.eval
    model.eval()
    wrapper = _Wrapper(model).eval()
    size = 256
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["image"], output_names=["classification", "mass"],
        dynamic_axes={"image": {0: "batch"}, "classification": {0: "batch"}, "mass": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # sanity check against torch
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    c, m = sess.run(None, {"image": dummy.numpy()})
    with torch.no_grad():
        tc, tm = wrapper(dummy)
    print("max |logit diff|:", float(np.abs(c - tc.numpy()).max()), "max |mass diff|:", float(np.abs(m - tm.numpy()).max()))
    print("saved", out_path)


if __name__ == "__main__":
    main()
