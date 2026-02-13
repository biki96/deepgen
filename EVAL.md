## Evaluation
For now we only provide inference code, please turn to the official repos of the benchmarks to calculate final performance.

### GenEval
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/gen_eval.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data
         --height 512 \
          --width 512 \
         --seed 42
```

