
cd ..

python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel.json --only_ig True --model_path Qwen/Qwen3-8B
python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel_insights_corrupted.json --only_ig True --model_path Qwen/Qwen3-8B
python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel_insights_filler_tokens.json --only_ig True --model_path Qwen/Qwen3-8B
python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel_insights_irrelevant.json --only_ig True --model_path Qwen/Qwen3-8B
python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel_irrelevant_trajectory.json --only_ig True --model_path Qwen/Qwen3-8B
python pipeline.py --full_data_path ../data/input_data/Qwen3-8B/expel_shuffle_trajectory.json --only_ig True --model_path Qwen/Qwen3-8B
