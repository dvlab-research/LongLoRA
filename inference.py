import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from llama_attn_replace import replace_llama_attn
from queue import Queue
from threading import Thread
import gradio as gr

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--question', type=str, default="")
    parser.add_argument('--material', type=str, default="")
    parser.add_argument('--material_title', type=str, default="")
    parser.add_argument('--material_type', type=str, default="material")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

def format_prompt(material, message, material_type="book", material_title=""):
    if material_type == "paper":
        prompt = f"Below is a paper. Memorize the material and answer my question after the paper.\n {material} \n "
    elif material_type == "book":
        material_title = ", %s"%material_title if len(material_title)>0 else ""
        prompt = f"Below is some paragraphs in the book{material_title}. Memorize the content and answer my question after the book.\n {material} \n "
    else:
        prompt = f"Below is a material. Memorize the material and answer my question after the material. \n {material} \n "
    message = str(message).strip()
    prompt += f"Now the material ends. {message}"

    return prompt

def read_txt_file(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(material, question, material_type="", material_title=None):
        material = read_txt_file(material.name)
        prompt = format_prompt(material, question, material_type, material_title)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        out = out.split(prompt)[1].strip()
        return out

    return response

def main(args):
    if args.flash_attn:
        replace_llama_attn()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=not args.flash_attn)

    output = respond(args.material, args.question, args.material_type, args.material_title)
    print("output", output)

if __name__ == "__main__":
    args = parse_config()
    main(args)
