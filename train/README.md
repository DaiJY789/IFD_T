# IFD_T Tool-RL Training (verl)

## 1) Build verl-format parquet

Use this to convert `image_path/label/forgery_types` parquet into the multi-turn + tool-calling RL schema expected by verl.

```bash
python /data/home/yxzhou/jydai/work/IFD_T/data/build_ifd_t_tool_rl_dataset.py \
  --input-dir /ssd2/jydai/data/IFD_T/ifd_t_105k_v1 \
  --output-dir /ssd2/jydai/data/IFD_T/ifd_t_105k_tool_rl_v1
```

Outputs:
- `train.parquet`
- `test.parquet`
- `eval.parquet`
- `eval.json`

## 2) Run GRPO training

```bash
bash /data/home/yxzhou/jydai/work/IFD_T/train/run_verl_ifd_t_tool_train.sh \
  --train-file /ssd2/jydai/data/IFD_T/ifd_t_105k_tool_rl_v1/train.parquet \
  --val-file /ssd2/jydai/data/IFD_T/ifd_t_105k_tool_rl_v1/test.parquet \
  --model-path /ssd2/jydai/model/Qwen3-VL-8B-Instruct \
  --gpus 2,3,4,5
```

Recommended env:
- `conda activate verl_jy`

## 3) Key files used by training

- Tool config: `/data/home/yxzhou/jydai/work/IFD_T/tools/verl_tool_config.yaml`
- Reward fn: `/data/home/yxzhou/jydai/work/IFD_T/reward/reward.py`
- Prompt files:
  - `/data/home/yxzhou/jydai/work/IFD_T/prompt/system_prompt.md`
  - `/data/home/yxzhou/jydai/work/IFD_T/prompt/user_prompt_template.md`

## 4) Logs and outputs

Default run output root:
- `/data/home/yxzhou/jydai/work/IFD_T/outputs/train_runs/<run_name>/`

Important paths:
- `logs/train_stdout.log`
- `logs/train_metrics.jsonl`
- `outputs/checkpoints/`
- `outputs/rollout_data/`
- `outputs/validation_data/`
