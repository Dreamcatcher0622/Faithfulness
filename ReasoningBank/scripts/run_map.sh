BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770/"
export WA_SHOPPING_ADMIN="$BASE_URL:7780/admin"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="https://www.openstreetmap.org"
export WA_HOMEPAGE="$BASE_URL:4399"

model_name="Qwen/Qwen3-32B"

export BROWSERGYM_HEADLESS=1

cd ..

python pipeline.py --website "map" --model_name "$model_name" --use_bank True

python pipeline.py --website "map" --model_name "$model_name" --use_bank False

types=("wo" "corrupt" "irrelevant" "filler_tokens")
# types=("filler_tokens")
for type in "${types[@]}"; do
    python pipeline.py --website "map" --model_name "$model_name" \
        --faithfulness_experiment "True" --insights_modification_type "$type"
done

