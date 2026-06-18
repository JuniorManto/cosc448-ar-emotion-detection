"""
this turns the pre-trained MLP from the EmoHeVRDB paper (fau_model.keras) into an onnx file (fau_model.onnx) so Unity's Inference Engine (i think formerly called Sentis) can load it and run it on the Quest Pro 
.onnx is just the file format Unity needs, the same way a .png is the format an image viewer needs

the model is a tiny 3-layer network (63 -> 128 -> 64 -> 7) so instead of installing a whole ML framework, i just read the weight numbers straight out of the .keras file (which is secretly a zip with an h5 weights file inside) and hand-build the onnx graph

the model (from inspection of the .keras):
- input: 63 face blendshape floats (the OVRFaceExpressions values, 0...1)
- layer1: Dense 128 + ReLU
- layer2: Dense 64 + ReLU
- layer3: Dense 7 + Softmax (7 outputs = the 7 class scores)
- outputs: [Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger]  (this exact order)

this model has no contempt and does have neutral which is different from my cosine classifier's label set 
argmax of the 7 outputs = the predicted emotion
"""

import sys
import zipfile
import numpy as np
import h5py
import onnx
from onnx import helper, numpy_helper, TensorProto

#the 7 output labels in the order the model was trained on (from the training notebook)
#i only keep this here so the optional self-test at the bottom can print readable names
LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]


def load_dense_weights(keras_path):
  #a .keras file is really a zip
  #the weights live inside it in model.weights.h5
  #i pull that out into memory and read every array out of it
  with zipfile.ZipFile(keras_path) as z:
    with z.open("model.weights.h5") as f:
      raw = f.read()

  #h5py needs a file-like thing so i wrap the bytes i just read
  import io
  arrays = []
  with h5py.File(io.BytesIO(raw), "r") as h:
  #walk the file and grab every dataset no matter how deep its buried
  #i match layers by shape later so i don't care what keras named the groups
  def collect(name, obj):
    if isinstance(obj, h5py.Dataset):
      arrays.append(np.array(obj))
    h.visititems(collect)

  #split into 2D kernels (the weight matrices) and 1D biases
  kernels = [a for a in arrays if a.ndim == 2]
  biases = [a for a in arrays if a.ndim == 1]

  #chain the layers together by shape - the first layer takes 63 inputs, and each layers output size is the next layers input size
  #this orders them correctly (63 -> 128, then 128 -> 64, then 64 -> 7) without trusting the files naming
  ordered = []
  #the models input is the 63 blendshapes
  in_dim = 63  
  used = [False] * len(kernels)
  for _ in range(len(kernels)):
    found = False
      for i, K in enumerate(kernels):
        if not used[i] and K.shape[0] == in_dim:
          #grab the matching bias (its length == this layers output size)
            out_dim = K.shape[1]
            bias = next(b for b in biases if b.shape[0] == out_dim)
            ordered.append((K.astype(np.float32), bias.astype(np.float32)))
            used[i] = True
            #next layers input is this layers output
            in_dim = out_dim 
            found = True
          break
        if not found:
          raise RuntimeError(f"couldn't find a layer taking {in_dim} inputs, shapes are off")
  #list of (kernel, bias) in forward order
  return ordered  


def build_onnx(layers, out_path):
#i build the graph node by node
#keras Dense does  y = relu(x @ kernel + bias)
#onnx Gemm op does exactly x @ B + C when transB = 0, and keras kernels are already stored as [in, out] so they drop straight in with no transposing
nodes = []
#the constant weight tensors baked into the file
inits = []  
prev = "input"
n = len(layers)
for i, (K, b) in enumerate(layers):
  wname, bname = f"W{i}", f"b{i}"
  inits.append(numpy_helper.from_array(K, name=wname))
  inits.append(numpy_helper.from_array(b, name=bname))
  gemm_out = f"gemm{i}"
  #Gemm x @ kernel + bias
  nodes.append(helper.make_node("Gemm", [prev, wname, bname], [gemm_out], alpha = 1.0, beta = 1.0, transB = 0))
  if i < n - 1:
    #hidden layers get a ReLU after them
    relu_out = f"relu{i}"
    nodes.append(helper.make_node("Relu", [gemm_out], [relu_out]))
    prev = relu_out
  else:
    #last layer gets a Softmax so the 7 outputs are nice 0...1 confidences
    nodes.append(helper.make_node("Softmax", [gemm_out], ["output"], axis=1))

  #declare the input (1 row, 63 values) and output (1 row, 7 scores)
  inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 63])
  out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(layers[-1][0][0])])
  graph = helper.make_graph(nodes, "fau_mlp", [inp], [out], initializer=inits)
        
  #opset 13 covers Gemm/Relu/Softmax and is well inside what Unity Inference Engine supports
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
  #a conservative IR version that Sentis/Inference Engine reads happily
  model.ir_version = 8
  #yells if i built something invalid
  onnx.checker.check_model(model)  
  onnx.save(model, out_path)


def main():
  keras_path = sys.argv[1] if len(sys.argv) > 1 else "fau_model.keras"
  out_path = "fau_model.onnx"
  layers = load_dense_weights(keras_path)
  shapes = " -> ".join([str(layers[0][0].shape[0])] + [str(K.shape[1]) for K, _ in layers])
  print(f"found {len(layers)} dense layers: {shapes}")
  build_onnx(layers, out_path)
  print(f"wrote {out_path}")

  #if onnxruntime is installed, run the onnx on a random face and check it matches plain numpy math on the same weights
  #if these dont match, the conversion is wrong (this needs nothing extra at the lab, its just a sanity check)
  try:
    import onnxruntime as ort
  except ImportError:
    print("(onnxruntime not installed, skipping the self-test, the file is still fine)")
      return
  x = np.random.default_rng(1).random((1, 63)).astype(np.float32)
  #numpy reference - same forward pass i described above
  h = x
  for i, (K, b) in enumerate(layers):
    h = h @ K + b
    if i < len(layers) - 1:
      #relu
      h = np.maximum(h, 0)
  #softmax
  ref = np.exp(h - h.max()) / np.exp(h - h.max()).sum() 
  got = ort.InferenceSession(out_path).run(None, {"input": x})[0]
  if np.allclose(ref, got, atol=1e-5):
      print(f"self-test PASSED, onnx matches numpy. predicted: {LABELS[int(got.argmax())]}")
  else:
    print("self-test FAILED, onnx output does not match numpy, do not use this file")

if __name__ == "__main__":
  main()
