cd ..
export CUDA_VISIBLE_DEVICES=3,4

python pipeline_rb.py --full_data_path ./inputs/input_prompt_with.json --only_ig True --model_path Qwen/Qwen3-14B 
# python pipeline_rb.py --full_data_path ./inputs/input_prompt_wo.json --only_ig True --model_path Qwen/Qwen3-14B 
python pipeline_rb.py --full_data_path ./inputs/input_prompt_corrupt.json --only_ig True --model_path Qwen/Qwen3-14B 
python pipeline_rb.py --full_data_path ./inputs/input_prompt_irrelevant.json --only_ig True --model_path Qwen/Qwen3-14B 
python pipeline_rb.py --full_data_path ./inputs/input_prompt_empty.json --only_ig True --model_path Qwen/Qwen3-14B 
python pipeline_rb.py --full_data_path ./inputs/input_prompt_filler_tokens.json --only_ig True --model_path Qwen/Qwen3-14B 

