## Joint training
### DeepGen_SCB

**Pretrain**

Multiple nodes are used for pretraining, so make sure you node rank, master address and master port has been written
into env variables `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
```shell
cd /path/to/deepgen
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=8 bash scripts/train_ddp.sh \
     configs/pretrain/deepgen_joint_pretrain_scb.py \
     --deepspeed deepspeed_zero2
```

**Finetune**

Modify the finetune config to set the path of your pretrained weights:
```python
model.pretrained_pth = 'path/to/the/pretrained/checkpoint.pth' 
# e.g., work_dirs/deepgen_joint_pretrain_scb/iter_200000.pth
```

```shell
cd /path/to/OpenUni
cd /path/to/deepgen
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=8 bash scripts/train_ddp.sh \
     configs/finetune/deepgen_joint_sft_scb.py \
     --deepspeed deepspeed_zero2
```
for last hidden condition for DiT variant are `deepgen_joint_sft` and `deepgen_joint_pretrain`


## train config


| Argument                     |Suggestions                                                     |
| ---------------------------- |  --------------------------------------------------------------- |
| accumulative_counts                         | equal to the multiple of the sum of the editing and generation ratios settings in dataset config ` repeats = ` |
| optim_type                   | `CustomAdamW`                 |               
| lr                   | `1e-4` for pretraining and `5e-5` for sft | 
| betas           | `32`                                        | 
| weight_decay          | `0.05`                                         | 
| max_norm | `70`                                        | 
| warmup_ratio     | `1.0 `                                       | 
| model.num_queries       | `128`                                       | 
| model.use_activation_checkpointing     | `False`                                       |
| model.freeze_transformer     |  whether DiT freeze during training, set `False`  when sft                                     | 
| model.lora_modules     | whether VLM utilize lora tuning, set `auto`  when sft                                       | 

## model config
