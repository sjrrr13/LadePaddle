"""
Run HumanEval. If LOAD_LADE, use Lookahead Decoding.
"""
from ast import comprehension
from base64 import decode
import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

import json
import time 
import os 
from tqdm import tqdm


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b", help="The directory of model.")
    parser.add_argument("--device", type=str, default="gpu", help="The device to use.")
    parser.add_argument("--max_len", type=int, default=256, help="The maximum length of the generated text.")
    parser.add_argument("--input_file", type=str, default="./Humaneval_Solution.jsonl", help="The input file.")
    parser.add_argument("--output_file", type=str, default="result.jsonl", help="The output file.")
    parser.add_argument("--sample", action="store_true", help="Whether to use sampling.")
    parser.add_argument("--topk", type=int, default=1, help="top_k parameter for generation")
    parser.add_argument("--topp", type=float, default=1.0, help="top_p parameter for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="top_p parameter for generation")
    # For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 (recommanded)
    parser.add_argument("--level", type=int, default=5, help="N-gram")
    parser.add_argument("--window", type=int, default=7, help="Window size")
    parser.add_argument("--guess", type=int, default=7, help="Guess set size")
    return parser


def parse_arguments():
    parser = get_parser()
    return parser.parse_args()


class Predictor:
    def __init__(self, args):
        if args is None:
            print("No arguments provided.")
            return
        
        self.model_name = args.model_name_or_path
        self.max_len = args.max_len
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="float16")
        self.model.eval()

    def preprocess(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors="pd")
        return model_inputs

    def infer(self, model_inputs):
        start = time.time()
        if self.args.sample:
            with paddle.no_grad():
                output = self.model.generate(
                    **model_inputs,
                    max_length=self.max_len,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="sampling",
                    temperature=self.args.temperature,
                    top_k=self.args.topk,
                    top_p=self.args.topp,
                )
        else:
            with paddle.no_grad():
                output = self.model.generate(
                    **model_inputs, 
                    max_length=self.max_len,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="greedy_search"
                )
        end = time.time()
        
        output = output[0]
        speed = output.shape[-1] / (end - start)
        return output, speed

    def postprocess(self, output):
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # hard code for humaneval
        output = '    ' + output.lstrip()
        if 'def' in output:
            idx = output.index('def')
            output = output[:idx]
        if '\n\n\n' in output:
            output = output.replace('\n\n\n', '\n')
        return output

    def predict(self, prompt):
        model_inputs = self.preprocess(prompt)
        output, speed = self.infer(model_inputs)
        output = self.postprocess(output)

        return output, speed, 0, []
        

class LadePredictor:
    def __init__(self, args):
        if args is None:
            print("No arguments provided.")
            return
        
        self.model_name = args.model_name_or_path
        self.max_len = args.max_len
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="float16")
        self.model.eval()

    def preprocess(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors="pd")
        return model_inputs

    def infer(self, model_inputs):
        prefix_len = model_inputs["input_ids"].shape[-1]

        start = time.time()
        if self.args.sample:
            with paddle.no_grad():
                output, compression, windows = self.model.generate(
                    **model_inputs, 
                    max_length=self.max_len, 
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="sampling", 
                    temperature=self.args.temperature,
                    top_k=self.args.topk, 
                    top_p=self.args.topp
                )
        else:
            with paddle.no_grad():
                output, compression, windows = self.model.generate(
                    **model_inputs, 
                    max_length=self.max_len, 
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="greedy_search"
                )
        end = time.time()
        
        output = output[0]
        speed = (output.shape[-1] - prefix_len) / (end - start)
        return output, speed, compression, windows

    def postprocess(self, output, prompt_len):
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        # hard code for humaneval
        output = output[prompt_len:]
        output = '    ' + output.lstrip()
        if 'def' in output:
            idx = output.index('def')
            output = output[:idx]
        if '\n\n\n' in output:
            output = output.replace('\n\n\n', '\n')
        return output

    def predict(self, prompt):
        model_inputs = self.preprocess(prompt)
        output, speed, compression, windows = self.infer(model_inputs)
        output = self.postprocess(output, len(prompt))

        return output, speed, compression, windows
    

if __name__ == "__main__":
    args = parse_arguments()
    paddle.device.set_device(args.device)
    use_lade = False

    if int(os.environ.get("LOAD_LADE", 0)):
        use_lade = True
        import lade 
        lade.augment_all()
        lade.config_lade(
            LEVEL=args.level, 
            WINDOW_SIZE=args.window, 
            GUESS_SET_SIZE=args.guess, 
            DEBUG=1, 
            POOL_FROM_PROMPT=True)
        print("LADE is enabled.")

    prompts = []
    taskid = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
            taskid.append(data['task_id'])

    if use_lade:
        predictor = LadePredictor(args)
    else:
        predictor = Predictor(args)

    outputs, speeds, comps, ws = [], [], [], []
    for prompt in tqdm(prompts, ncols=100):
        model_inputs = predictor.preprocess(prompt)
        output, speed, compression, windows = predictor.predict(prompt)  # compression is 0 if not using LADE
        outputs.append(output)
        speeds.append(speed)
        comps.append(compression)
        ws.append(windows)

    with open(args.output_file, "w") as f:
        for i in range(len(prompts)):
            data = {
                "task_id": taskid[i],
                "pred_output": outputs[i],
                "speed": speeds[i],
                "compression": comps[i],
                "windows": ws[i]
            }
            f.write(json.dumps(data) + "\n")
