python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name generic_child_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json

python run_eval.py \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway \
     --system_prompt_name duolingo_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json

python run_eval.py   \
     --model_name llama-3.1-8b-instruct \
     --provider scaleway   \
     --system_prompt_name generic_child_tutor_v1   \
     --prompt_pack_path prompt_packs/minorbench_hateful.json   \
     --guardrail_model ibm-granite/granite-guardian-3.2-3b-a800m   \
     --guardrail_provider local_hf \
     --safety_judge_provider local_hf \
     --safety_judge_model meta-llama/Llama-3.2-1B-Instruct
     # --safety_judge_model nya
