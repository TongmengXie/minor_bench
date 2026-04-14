source .minor/bin/activate

# Hateful-only baseline with local_hf judge.
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name assistant_child_safety_v2 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# Hateful + safe controls (variantized) with local_hf judge.
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name assistant_child_safety_v2 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --extra_dataset_paths datasets/true_negative_safe_sensitive_variants.jsonl \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# Fair-comparison Granite row (shared policy summary + local_hf judge).
python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name assistant_child_safety_v2 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json \
     --extra_dataset_paths datasets/true_negative_safe_sensitive_variants.jsonl \
     --guardrail_id granite_guardian_3_2_compare_v2 \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct

# judge_only
python run_eval.py --judge_only --judge_only_path runs/evals/llama-31-8b-instruct/assistant_child_safety_v2/<timestamp> --safety_judge_model meta-llama/Llama-3.2-1B-Instruct --safety_judge_provider local_hf

# Fair external-guardrail comparison matrix (rows = assistant-only vs assistant + specific guardrail, columns = variants).
python run_matrix_eval.py --matrix_config matrix_configs/hateful_guardrail_compare_matrix_assistant_policy_v2.yaml

# Legacy matrices kept for historical comparison only.
# python run_matrix_eval.py --matrix_config matrix_configs/hateful_guardrail_matrix.yaml
# python run_matrix_eval.py --matrix_config matrix_configs/hateful_guardrail_matrix_with_duolingo.yaml
