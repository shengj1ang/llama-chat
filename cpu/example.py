# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))


    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    print('Setup parallel complete!')
    # torch.cuda.set_device(local_rank)

    num_cpus = torch.get_num_threads()
    torch.set_num_threads(num_cpus)
    
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str="../models/7B",
    tokenizer_path: str="../models/tokenizer.model",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
    
    #print("")
    while 1:
        print('-' * 30)
        while 2:
                x=input("Prompts: ")
                if x!="":
                        break
                else:
                        print("Empty Prompt Is Not Allowed! Try it again.")
                
        prompts=[x]
        #print(prompts)
        #print("LLaMA: "+x+" ", end='\r')
        sys.stdout.write(f"LLaMA: {x} ")
        sys.stdout.flush()
        t1=time.time()
        #results = generator.generate(prompts, max_gen_len=512, temperature=temperature, top_p=top_p)
        results = generator.generate(prompts, max_gen_len=60, temperature=temperature, top_p=top_p)
        print(f"\nTime Consumed: {str(time.time()-t1)} s")
        #print(results)
        #Final Output
        #for i in results:
        #    print(i)
        print('-' * 30)

    '''
print("break-point-1")
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    # results = generator.generate(
    #     prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    # )
print("break-point-2")
    results = generator.generate(
        prompts, max_gen_len=512, temperature=temperature, top_p=top_p
    )
print("break-point-3")
    for result in results:
        print(result)
        print("\n==================================\n")
print("break-point-4")
'''
if __name__ == "__main__":
    fire.Fire(main)
    