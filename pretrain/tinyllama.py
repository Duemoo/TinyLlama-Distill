import glob
import math
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
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.knowledge_distillation_loss import KDLoss
import random
from lit_gpt import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_name = "tiny_LLaMA_120M"
name = "tinyllama_120M"
out_dir = Path("out") / name

# Hyperparameters
num_of_devices = 8
global_batch_size = 512
learning_rate = 4e-4
micro_batch_size = 16
max_step = 715256 * 2
warmup_steps = 2000
log_step_interval = 10
eval_iters = 100
save_step_interval = 5000
eval_step_interval = 200


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps




max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train_slim", 1.0)
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# automatically make version_XX directory in out_dir
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger(entity="lklab_kaist", project="tinyllama_distill")


def setup(
    num_devices: int = 8,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    corruption_rate: Optional[float] = 0.001,
) -> None:
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    print(f"num_devices: {num_devices}")

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

    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume, corruption_rate)


def main(fabric, train_data_dir, val_data_dir, resume, corruption_rate):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    student_config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=student_config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {student_config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        student_model = GPT(student_config)
        student_model.apply(partial(student_model._init_weights ,n_layer=student_config.n_layer))
        
        with torch.no_grad():
            new_wte = torch.zeros_like(student_model.transformer.wte.weight)
            new_wte = torch.normal(mean=new_wte, std=math.sqrt(2.0 / 5 / new_wte.size(1)))
            student_model.transformer.wte.weight.copy_(new_wte)
            
            new_lm = torch.zeros_like(student_model.lm_head.weight)
            new_lm = torch.normal(mean=new_lm, std=math.sqrt(2.0 / 5 / new_lm.size(1)))
            student_model.lm_head.weight.copy_(new_lm)
    teacher_model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480k-1T", use_flash_attention_2=True, torch_dtype=torch.bfloat16)
    print(f"teacher model {teacher_model} can generate : {teacher_model.can_generate()}")
    teacher_model.eval()

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(student_model):,}")

    student_model = fabric.setup(student_model)
    teacher_model = fabric.setup(teacher_model)
    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"student_model": student_model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
        print(f"\n\nfind last checkpoint : {resume}")
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, teacher_model, train_dataloader, val_dataloader, monitor, resume, corruption_rate)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, teacher_model, train_dataloader, val_dataloader, monitor, resume, corruption_rate):
    student_model = state["student_model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, student_model, teacher_model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(student_model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, student_model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = KDLoss(setting="train")
    corruption_check = 0
    
    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        # input_ids : (batch_size, block_size)
        input_ids = train_data[:, 0 : student_model.config.block_size].contiguous()
        current_batch_size = train_data.shape[0]
        
        corruption_check += current_batch_size
        if (corruption_check // (1 / corruption_rate)) > 0:
            # Corrupt label
            remove_idx = int(current_batch_size - (corruption_check % (1 / corruption_rate)) - 1)
            teacher_input_ids = torch.cat([input_ids[:remove_idx, :], input_ids[remove_idx+1:, :]], dim=0)
            corruption_check = int(corruption_check % (1 / corruption_rate))
        else:
            teacher_input_ids = input_ids
        with torch.no_grad():
            teacher_output = teacher_model(teacher_input_ids)["logits"]
            
        targets = train_data[:, 1 : student_model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(student_model, enabled=is_accumulating):
            logits = student_model(input_ids)
            # Intend to 
            if logits.shape[0] != teacher_output.shape[0]:
                teacher_output = torch.cat([teacher_output[:remove_idx, :], torch.rand(1, teacher_output.shape[1], teacher_output.shape[2], device=teacher_output.device), teacher_output[remove_idx:, :]], dim=0)
                assert teacher_output.shape == logits.shape
            loss = loss_func(pred=logits, target=targets, teacher_pred=teacher_output, T=2, alpha=1)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(student_model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

            
            
            
        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            
            t0 = time.perf_counter()
            val_kldiv_loss, val_ce_loss, val_kd_loss = validate(fabric, student_model, teacher_model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val ce loss {val_ce_loss:.4f}, val kd loss {val_kd_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_kl_divergence_loss": val_kldiv_loss.item(), "total_tokens":  student_model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.log_dict({"metric/val_cross_entropy_loss": val_ce_loss.item(), "total_tokens":  student_model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_ce_loss.item()), "total_tokens":  student_model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.log_dict({"metric/val_knowledge_distillation": val_kd_loss.item(), "total_tokens":  student_model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size},state["step_count"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"{name}-step_{state['step_count']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, student_model: torch.nn.Module, teacher_model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    student_model.eval()

    ce_losses = torch.zeros(eval_iters, device=fabric.device)
    kd_losses = torch.zeros(eval_iters, device=fabric.device)
    kldiv_losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : student_model.config.block_size].contiguous()
        # generation config check
        teacher_output = teacher_model(input_ids)["logits"]
        targets = val_data[:, 1 : student_model.config.block_size + 1].contiguous()
        logits = student_model(input_ids)
        # loss = chunked_cross_entropy(logits, teacher_output, chunk_size=0)
        loss_func = KDLoss(setting="validation")
        kldiv_loss, ce_loss, kd_loss = loss_func(pred=logits, target=targets, teacher_pred=teacher_output, T=2, alpha=1)
        
        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        kldiv_losses[k] = kldiv_loss.item()
        ce_losses[k] = ce_loss.item()
        kd_losses[k] = kd_loss.item()
    
    kldiv_out = kldiv_losses.mean()
    ce_out = ce_losses.mean()
    kd_out = kd_losses.mean()

    student_model.train()
    return (kldiv_out, ce_out, kd_out)


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
