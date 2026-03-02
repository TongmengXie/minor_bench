source .minor/bin/activate

# Hateful-only baseline with local_hf judge.
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name generic_child_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# Hateful + safe controls (variantized) with local_hf judge.
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name generic_child_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --extra_dataset_paths datasets/true_negative_safe_sensitive_variants.jsonl \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# Guardrail row (local_hf guardrail + local_hf judge).
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name generic_child_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --extra_dataset_paths datasets/true_negative_safe_sensitive_variants.jsonl \
     --guardrail_model ibm-granite/granite-guardian-3.2-3b-a800m \
     --guardrail_provider local_hf \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# judge_only
python run_eval.py --judge_only --judge_only_path runs/evals/llama-31-8b-instruct/generic_child_tutor_v1/20260301_084212 --safety_judge_model meta-llama/Llama-3.2-1B-Instruct --safety_judge_provider local_hf

# Guardrail x mutation matrix (rows = guardrail/system combinations, columns = variants).
python run_matrix_eval.py --matrix_config matrix_configs/hateful_guardrail_matrix.yaml

# Optional full matrix including Duolingo rows.
# python run_matrix_eval.py --matrix_config matrix_configs/hateful_guardrail_matrix_with_duolingo.yaml
