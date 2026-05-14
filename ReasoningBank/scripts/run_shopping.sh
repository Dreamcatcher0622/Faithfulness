BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770/"
export WA_SHOPPING_ADMIN="$BASE_URL:7780/admin"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"

model_name="Qwen/Qwen3-32B"

export BROWSERGYM_HEADLESS=1

cd ..

docker stop shopping
docker rm shopping
docker run --name shopping -p 7770:80 -d shopping_final_0712
sleep 180
docker start shopping
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

python pipeline.py --website "shopping" --model_name "$model_name" --use_bank True
docker stop shopping
docker rm shopping
docker run --name shopping -p 7770:80 -d shopping_final_0712
sleep 180
docker start shopping
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

python pipeline.py --website "shopping" --model_name "$model_name" --use_bank False
docker stop shopping
docker rm shopping
docker run --name shopping -p 7770:80 -d shopping_final_0712
sleep 180
docker start shopping
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush


types=("wo" "corrupt" "irrelevant" "filler_tokens")
# types=("filler_tokens")
for type in "${types[@]}"; do
    python pipeline.py --website "shopping" --model_name "$model_name" \
        --faithfulness_experiment "True" --insights_modification_type "$type"
    docker stop shopping
    docker rm shopping
    docker run --name shopping -p 7770:80 -d shopping_final_0712
    sleep 180
    docker start shopping
    docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
    docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
    docker exec shopping /var/www/magento2/bin/magento cache:flush
done
