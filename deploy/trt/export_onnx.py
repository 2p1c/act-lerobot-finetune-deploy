import os
import collections

import torch
import onnx

from lerobot.policies.act.modeling_act import ACTPolicy

POLICY = os.path.expanduser(
    "~/outputs/act/so101-pick-cube-clean/checkpoints/last/pretrained_model"
)
ONNX_PATH = "deploy/trt/act.onnx"

class ActForward(torch.nn.Module):
    def __init__(self, act_model):
        super().__init__()
        self.model = act_model

    def forward(self, state, wrist, front):
        batch = {
            "observation.state": state, 
            "observation.images": [wrist, front],
        }
        actions = self.model(batch)[0]
        return actions

def main():
    policy = ACTPolicy.from_pretrained(POLICY)
    policy.to("cpu")
    policy.eval()

    wrapper = ActForward(policy.model)
    wrapper.eval()

    state = torch.randn(1, 6)
    wrist = torch.randn(1, 3, 640, 480)
    front = torch.randn(1, 3, 640, 480)

    with torch.no_grad():
        out = wrapper(state, wrist, front)
    print("Pytorch 输出形状:", tuple(out.shape))

    with torch.no_grad():
        torch.onnx.export(wrapper, (state, wrist, front), ONNX_PATH, opset_version=17, dynamo=False, input_names=["state", "wrist", "front"], ouput_names=["actions"],)
    print("已导出:", ONNX_PATH)

    onnx.checker.check_model(ONNX_PATH)
    print("checker 通过")
    m = onnx.load(ONNX_PATH)
    print("节点总数:", len(m.graph.node))
    counter = collections.Counter(n.op_type for n in m.graph.node)
    print("最多的算子:", counter.most_common(10))
    print("文件大小: %.1f MB" % (os.path.getsize(ONNX_PATH) / 1e6))

    for i in m.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        print("输入", i.name, dims)
    for o in m.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print("输出", o.name, dims)

if __name__ == "__main__":
    main()

