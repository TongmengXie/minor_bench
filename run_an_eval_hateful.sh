python run_eval.py \
     --model_name gpt-4o-mini \
     --provider openai_api \
     --system_prompt_name generic_child_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json

python run_eval.py \
     --model_name gpt-4o-mini \
     --provider openai_api \
     --system_prompt_name duolingo_tutor_v1 \
     --prompt_pack_path prompt_packs/minorbench_hateful.json
