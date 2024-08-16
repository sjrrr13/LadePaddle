from decimal import MAX_EMAX
import json
from lib2to3.pgen2.tokenize import tokenize
from pyexpat import model
from typing_extensions import Self
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import os 
from tqdm import tqdm
import torch.distributed as dist 


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="The directory of model.")
    parser.add_argument("--torch_device", type=str, default="cuda", help="The torch device to use.")
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
            exit()
        
        self.model_name = args.model_name_or_path
        self.torch_device = args.torch_device
        self.max_len = args.max_len
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            trust_remote_code=True, 
            device_map='auto')
        self.model.tokenizer = self.tokenizer
        self.model.eval()

    def preprocess(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.torch_device)
        return model_inputs

    def infer(self, model_inputs):
        output = self.model.generate(**model_inputs, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)    # warm up
        torch.cuda.synchronize()
        start = time.time()
        if self.args.sample:
            output = self.model.generate(**model_inputs, max_new_tokens=self.max_len, do_sample=True, 
                                        top_k=self.args.topk, top_p=self.args.topp, temperature=self.args.temperature, pad_token_id=self.tokenizer.eos_token_id)
        else:
            output = self.model.generate(**model_inputs, max_new_tokens=self.max_len, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        end = time.time()
        
        speed = (output.numel() - model_inputs['input_ids'].numel()) / (end - start)
        return output, speed

    def postprocess(self, output, prefix_len):
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # hard code for humaneval
        output = output[prefix_len:]
        output = '    ' + output.lstrip()
        if 'def' in output:
            idx = output.index('def')
            output = output[:idx]
        if '\n\n\n' in output:
            output = output.replace('\n\n\n', '\n')
        return output

    def predict(self, prompt):
        inputs = self.preprocess(prompt)
        output, speed = self.infer(inputs)
        output = self.postprocess(output, len(prompt))

        return output, speed
        

if __name__ == "__main__":
    args = parse_arguments()

    if int(os.environ.get("LOAD_LADE", 0)):
        import lade 
        lade.augment_all()
        lade.config_lade(
            LEVEL=args.level, 
            WINDOW_SIZE=args.window, 
            GUESS_SET_SIZE=args.guess, 
            DEBUG=1, 
            POOL_FROM_PROMPT=True,
            DIST_WORKERS=int(os.environ.get("DIST_WORKERS", 1)))
        print("LADE is enabled.")

    prompts = []
    taskid = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
            taskid.append(data['task_id'])

    predictor = Predictor(args)
    outputs, speeds = [], []
    for idx, prompt in tqdm(enumerate(prompts)):
        output, speed = predictor.predict(prompt)
        outputs.append(output)
        speeds.append(speed)
    
    with open(args.output_file, "w") as f:
        for i in range(len(prompts)):
            data = {
                "task_id": taskid[i],
                "pred_output": outputs[i],
                "speed": speeds[i]
            }
            f.write(json.dumps(data) + "\n")
