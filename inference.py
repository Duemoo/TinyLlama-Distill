import glob
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
import torch.nn.functional as F
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.knowledge_distillation_loss import KDLoss
import random
from lit_gpt import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


model_name = "tiny_LLaMA_120M"
name = "tinyllama_120M_corr1e-3"
pretrained_model_dir = Path("/mnt/sda/hoyeon/TinyLlama-Distill/out") / name

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    num_devices: int = 1,
    precision: Optional[str] = None,
    tpu: bool = False,
    ood_threshold: float = 10,
    hall_threshold: float = 10,
    mode: str = "output_check"
) -> None:
    
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if num_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            num_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision, loggers=[])
    fabric.print(hparams)
    main(fabric, mode, ood_threshold, hall_threshold)
    
    
def main(fabric, mode, ood_threshold, hall_threshold):

    student_config = Config.from_name(model_name)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {student_config.__dict__}")
    t0 = time.perf_counter()

    student_model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, use_flash_attention_2=True, torch_dtype=torch.bfloat16).to("cuda")
    teacher_model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480k-1T", use_flash_attention_2=True, torch_dtype=torch.bfloat16).to("cuda")
    print(f"teacher model {teacher_model} can generate : {teacher_model.can_generate()}")

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(student_model):,}")

    inference(fabric, student_model, teacher_model, ood_threshold, hall_threshold)


def top_p_sampling(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')  # Set logits of unwanted tokens to negative infinity
    return logits


def generate(model, start_input_ids, max_length, top_p=0.9):
    model.eval()  # Set the model to evaluation mode

    input_ids = start_input_ids

    for _ in range(max_length):
        with torch.no_grad():  # No need to track gradients
            # Get the logits for the next token
            logits = model(input_ids)[:,-1,:]  # Only consider the last step

            # Apply top-p sampling to the logits
            filtered_logits = top_p_sampling(logits, top_p=top_p)

            # Sample the next token from the filtered distribution
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
            print(f"probabilities: {probabilities}")
            print(f"probabilities: {probabilities.shape}")
            assert False
            next_token = torch.multinomial(probabilities, 1)

            # Append the predicted next token to the input IDs to use as input for the next iteration
            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Optionally, if your model generates an end-of-sequence token, you could break the loop if that token is generated

    return input_ids


def inference(fabric, student_model, teacher_model, ood_threshold, hall_threshold):
    student_model.eval()
    teacher_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480k-1T")
    
    while(1):
        input_prompt = input("Write prompt: ")
        input_ids = torch.unsqueeze(torch.tensor(tokenizer(input_prompt)["input_ids"]), 0).to("cuda")
        # print(input_ids.shape)
        
        generation_kwargs = {
            'do_sample': False,
            'num_beams': 1,
            'max_new_tokens': 64
        }
        
        teacher_output_ids = teacher_model.generate(input_ids, **generation_kwargs)
        student_output_ids = student_model.generate(input_ids, **generation_kwargs)

        # student_output_logit = student_model(input_ids)
        # student_output_ids = torch.argmax(torch.nn.functional.softmax(student_output_logit, dim=2), dim=2)
        # student_output_ids = generate(student_model, input_ids, max_length=64)
        # print(f"student_output_ids : {student_output_ids}")
        
        teacher_output = tokenizer.batch_decode(teacher_output_ids)
        student_output = tokenizer.batch_decode(student_output_ids)
        print('\n')
        print(f"teacher_output: {teacher_output}\nstudent_output: {student_output}")
        
        print('\n\n')
        print('-'*50)
        print('\n')
        
        
if __name__=="__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)