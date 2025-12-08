import numpy as np
import tritonclient.http as httpclient

# Adjust if you changed ports / host
TRITON_URL = "localhost:8000"
MODEL_NAME = "simple"

client = httpclient.InferenceServerClient(url=TRITON_URL)

# Example input shaped [1, 1, 28, 28]
input_data = np.random.rand(1, 1, 28, 28).astype(np.float32)

inputs = []
inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
inputs[0].set_data_from_numpy(input_data)

outputs = []
outputs.append(httpclient.InferRequestedOutput("output"))

response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

result = response.as_numpy("output")
print("Output shape:", result.shape)
print("Output:", result)
