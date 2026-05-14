BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770/"
export WA_SHOPPING_ADMIN="$BASE_URL:7780/admin"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"



export BROWSERGYM_HEADLESS=1

cd ..

docker stop gitlab
docker rm gitlab
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
sleep 180
docker start gitlab
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

python pipeline.py --website "gitlab" --model_name 'gemini-2.5-flash' --use_bank False
docker stop gitlab
docker rm gitlab
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
sleep 180
docker start gitlab
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

types=("wo" "corrupt" "irrelevant" "filler_tokens")
# types=("irrelevant" "filler_tokens")
for type in "${types[@]}"; do
    python pipeline.py --website "gitlab" --model_name 'gemini-2.5-flash' \
        --faithfulness_experiment "True" --insights_modification_type "$type"
    docker stop gitlab
    docker rm gitlab
    docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
    sleep 180
    docker start gitlab
    docker exec gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
    docker exec gitlab gitlab-ctl reconfigure
done

# python pipeline.py --website "gitlab" --model_name 'genimi-2.5-flash' \
#     --faithfulness_experiment "True" --fewshot_modification_type "shuffle"
# docker stop gitlab
# docker rm gitlab
# docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
# sleep 180
# docker start gitlab
# docker exec gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
# docker exec gitlab gitlab-ctl reconfigure