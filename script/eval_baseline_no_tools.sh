python /data/home/yxzhou/jydai/work/IFD_T/eval/eval_baseline_no_tools.py \
  --chat-endpoint http://127.0.0.1:8010/v1/chat/completions \
  --model-name Qwen3vl8b \
  --eval-input /ssd2/jydai/data/IFD_T/ifd_t_105k_v1/eval/eval.parquet \
  --output-dir /data/home/yxzhou/jydai/work/IFD_T/eval/output/baseline_no_tools \
  --system-prompt /data/home/yxzhou/jydai/work/IFD_T/prompt/system_prompt_baseline_no_tools.md\
  --user-template /data/home/yxzhou/jydai/work/IFD_T/prompt/user_prompt_template_baseline_no_tools.md \
  --max-samples 0 \
  --temperature 0.1 \
  --max-tokens 4096 \
  --log-every 10
  