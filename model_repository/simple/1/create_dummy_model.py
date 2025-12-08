import onnx
from onnx import helper, TensorProto, OperatorSetIdProto

# Input: [1, 1, 28, 28]
input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 1, 28, 28]
)

# Output: [1, 1, 28, 28]
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 1, 28, 28]
)

# Identity node: output = input
node = helper.make_node(
    "Identity",
    inputs=["input"],
    outputs=["output"],
    name="identity_node",
)

graph = helper.make_graph(
    [node],
    "SimpleIdentityGraph",
    [input_tensor],
    [output_tensor],
)

model = helper.make_model(
    graph,
    producer_name="simple_identity",
)

# 🔥 Force a low IR version that Triton supports
model.ir_version = 7  # <= 10 is required

# 🔧 Set opset to something safe (<= 11)
opset = OperatorSetIdProto()
opset.domain = ""
opset.version = 11
model.opset_import.clear()
model.opset_import.append(opset)

onnx.checker.check_model(model)

onnx.save(model, "model.onnx")
print("Saved model.onnx with IR version:", model.ir_version)

