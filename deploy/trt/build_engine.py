import argparse
import sys
import time 

import tensorrt as trt

def main():
  parser_args = argparse.ArgumentParser()
  parser_args.add_argument("--onnx", required=True)
  parser_args.add_argument("--engine", required=True)
  parser_args.add_argument("--fp16", action="store_true")
  parser_args.add_argument("--verbose", action="store_true")
  args = parser_args.parse_args()

  level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
  logger = trt.Logger(level)
  builder = trt.Builder(logger)
  network = builder.create_network(0)

  parser = trt.OnnxParser(network, logger)
  if not parser.parse_from_file(args.onnx):
    for i in range(parser.num_errors):
      print("ONNX 解析错误:", parser.get_error(i))
    sys.exit(1)

  for i in range(network.num_inputs):
      t = network.get_input(i)
      print("输入", t.name, tuple(t.shape), t.dtype)
  for i in range(network.num_outputs):
      t = network.get_output(i)
      print("输出", t.name, tuple(t.shape), t.dtype)
  print("TensorRT 网络层数（融合前）:", network.num_layers)

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
  if args.fp16:
    config.set_flag(trt.BuilderFlag.FP16)
  config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
  print("开始构建，FP16 =", args.fp16)
  t0 = time.perf_counter()
  serialized = builder.build_serialized_network(network, config)
  t1 = time.perf_counter()
  if serialized is None:
    print("构建失败")
    sys.exit(1)
  print("构建耗时： %.1f s" % (t1 - t0))

  with open(args.engine, "wb") as f:
    f.write(serialized)
  print("已保存：", args.engine, "%.1f MB" % (serialized.nbytes / 1e6))

if __name__ == "__main__":
  main()


  
  