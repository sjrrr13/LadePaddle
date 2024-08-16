import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
import time 
import os 

if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
    lade.config_lade(LEVEL=5, WINDOW_SIZE=3, GUESS_SET_SIZE=3, DEBUG=1, POOL_FROM_PROMPT=True)

model_name = "meta-llama/Llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")

prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
model_inputs = tokenizer(prompt, return_tensors="pd")
prompt_len = (paddle.numel(model_inputs['input_ids'])).item()

#warm up
# with paddle.no_grad():
#     greedy_output = model.generate(**model_inputs, max_length=1, decode_strategy="sampling", temperature=0.6, top_k=50, top_p=0.9)
#end warm up

# ** Greedy search **
# paddle.device.synchronize()
# t0 = time.time()
# with paddle.no_grad():
#     output, compression = model.generate(**model_inputs, max_length=256, decode_strategy="greedy_search")
# paddle.device.synchronize()
# t1 = time.time()

# ** Greedy search without lade **
paddle.device.synchronize()
t0 = time.time()
with paddle.no_grad():
    output, compression, windows = model.generate(**model_inputs, max_length=256, decode_strategy="greedy_search")
paddle.device.synchronize()
t1 = time.time()

# ** Sampling **
# paddle.device.synchronize()
# t0 = time.time()
# with paddle.no_grad():
#     output, compression = model.generate(**model_inputs, max_length=256, decode_strategy="sampling", temperature=0.8, top_p=0.95)
# paddle.device.synchronize()
# t1 = time.time()

output = output[0]
out_text = tokenizer.decode(output, skip_special_tokens=False)

# gen_len = (paddle.numel(output)).item()
# speed = (gen_len - prompt_len) / (t1 - t0)
speed = 0
compression = 0
print(f"========== output ==========\n{out_text}\n====================\n{speed} tokens/s\tcompression: {compression}\n====================\nwindows: {windows}")

#python minimal.py #44 tokens/s
#LOAD_LADE=1 USE_LADE=1 python minimal.py #74 tokens/s, 1.6x throughput without changing output distribution!
