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

docker stop shopping_admin
docker rm shopping_admin
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
sleep 180
docker start shopping_admin
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

python pipeline.py --website "shopping_admin" --model_name "$model_name" --use_bank True
docker stop shopping_admin
docker rm shopping_admin
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
sleep 180
docker start shopping_admin
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush


python pipeline.py --website "shopping_admin" --model_name "$model_name" --use_bank False
docker stop shopping_admin
docker rm shopping_admin
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
sleep 180
docker start shopping_admin
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

types=("wo" "corrupt" "irrelevant" "filler_tokens")
types=("irrelevant" "filler_tokens")
for type in "${types[@]}"; do
    python pipeline.py --website "shopping_admin" --model_name "$model_name" \
        --faithfulness_experiment "True" --insights_modification_type "$type"
    docker stop shopping_admin
    docker rm shopping_admin
    docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
    sleep 180
    docker start shopping_admin
    docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780" # no trailing slash
    docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
    docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
done