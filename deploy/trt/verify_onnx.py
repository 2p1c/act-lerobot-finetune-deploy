import os 
import numpy as np
import torch
import onnxruntime as ort

from lerobot.policies.act.modeling_act import ACTPolicy
from export_onnx import ActForward, POLICY, ONNX_PATH

REF_PATH = "deploy/trt/ref.npz"

def main():
  torch.manual_seed(0)
  state = torch.randn(1, 6)
  wrist = torch.randn(1, 3, 640, 480)
  front = torch.randn(1, 3, 640, 480)

  policy = ACTPolicy.from_pretrained(POLICY)
  policy.to("cpu")
  policy.eval()
  wrapper = ActForward(policy.model).eval()

  with torch.no_grad():
    out_torch = wrapper(state, wrist, front).numpy()

  sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

  feeds = {
    "state": state.numpy(),
    "wrist": wrist.numpy(),
    "front": front.numpy(),
  }

  out_onnx = sess.run(["actions"], feeds)[0]

  diff = np.abs(out_torch - out_onnx)
  print("输出形状  torch:", out_torch.shape, " onnx:", out_onnx.shape)
  print("max  abs diff: %.2e" % diff.max())
  print("mean abs diff: %.2e" % diff.mean())
  print("输出数值范围: [%.3f, %.3f]" % (out_torch.min(), out_torch.max()))

  if diff.max() < 1e-3:
      print("✅ ONNX 导出正确")
  else:
      print("❌ 差得太多，导出有问题")

  # ========== ⑤ 存下来，第 5 步拿来和 TensorRT 比 ==========
  np.savez(
      REF_PATH,
      state=state.numpy(),
      wrist=wrist.numpy(),
      front=front.numpy(),
      actions=out_torch,
  )
  print("参考数据已保存:", REF_PATH)


if __name__ == "__main__":
  main()