import json
import csv
import os
import numpy as np
import requests
import json


def generate_embeddings_for_intents():
    """
    读取config_files文件夹中的0.json-811.json文件，
    提取intent字段，生成嵌入并保存到CSV文件
    """
    url = os.getenv("OPENAI_API_BASE")
    Skey = os.getenv("OPENAI_API_KEY")

    # 存储所有嵌入的列表
    all_embeddings = []
    # 存储对应的文件信息
    file_info = []
    
    # 处理0.json到811.json
    for i in range(812):  # 0到811共812个文件
        filename = f"../config_files/{i}.json"
        
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在，跳过")
            continue
            
        try:
            # 读取JSON文件
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取intent字段
            intent = data.get('intent')
            if not intent:
                print(f"警告: 文件 {filename} 中没有intent字段")
                continue
            
            print(f"处理文件 {filename}: {intent[:50]}...")
            
            # 生成嵌入
            # response = client.models.embed_content(
            #     model="gemini-embedding-001",
            #     contents=intent,
            #     config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            # )
            payload = json.dumps({
                "model": "text-embedding-ada-002",
                "input": f"{intent}"
            })

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {Skey}',
            }
            for i in range(3):
                response = requests.request("POST", url, headers=headers, data=payload)
                if response.status_code == 200:
                    break
            data = response.json()
            # print(data)
            
            # print(data['data'][0]['embedding'])
            # 获取嵌入向量
            embedding = data['data'][0]['embedding']
            # embedding = response.embeddings[0].values
            # print(embedding)
            
            if embedding:
                # 存储嵌入和文件信息
                all_embeddings.append(embedding)
                file_info.append({
                    'file_id': i,
                    'intent': intent,
                    'filename': filename
                })
            
        except Exception as e:
            print(f"错误: 处理文件 {filename} 时出错: {e}")
            continue
    
    # 将嵌入保存到CSV文件
    if all_embeddings:
        output_filename = "intent_embeddings.csv"
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for embedding in all_embeddings:
                writer.writerow(embedding)
            
            # # 写入表头（可选：包含文件ID信息）
            # header = ['file_id'] + [f'dim_{j}' for j in range(len(all_embeddings[0]))]
            # writer.writerow(header)
            
            # # 写入每一行的嵌入数据
            # for i, embedding in enumerate(all_embeddings):
            #     row = [file_info[i]['file_id']] + embedding
            #     writer.writerow(row)
        
        print(f"\n完成! 共处理 {len(all_embeddings)} 个文件的嵌入")
        print(f"嵌入数据已保存到: {output_filename}")
        
        # 同时保存文件信息到单独的CSV（可选）
        info_filename = "file_info.csv"
        with open(info_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_id', 'filename', 'intent'])
            for info in file_info:
                writer.writerow([info['file_id'], info['filename'], info['intent']])
        
        print(f"文件信息已保存到: {info_filename}")
        
        return all_embeddings, file_info
    else:
        print("错误: 没有成功生成任何嵌入")
        return None, None

def load_embeddings_from_csv(filename="intent_embeddings.csv"):
    """
    从CSV文件加载嵌入数据
    """
    embeddings = []
    file_ids = []
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 跳过表头
        
        for row in reader:
            file_ids.append(int(row[0]))
            embedding = [float(x) for x in row[1:]]
            embeddings.append(embedding)
    
    return embeddings, file_ids

# 运行主函数
if __name__ == "__main__":
    # 检查config_files文件夹是否存在
    if not os.path.exists("config_files"):
        print("错误: config_files文件夹不存在")
        print("请确保脚本与config_files文件夹在同一目录下")
    else:
        # 生成嵌入
        embeddings, file_info = generate_embeddings_for_intents()