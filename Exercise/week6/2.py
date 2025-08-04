import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import open_clip  # 使用 open_clip 替代 clip

# 配置设置
class Config:
    image_dir = "images"  # 图像目录
    feature_db = "image_features.npy"  # 特征向量数据库文件
    path_db = "image_paths.txt"  # 图像路径数据库文件
    top_k = 5  # 检索返回的最相似图像数量
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    model_name = "ViT-B-32"  # CLIP模型版本
    pretrained = "laion2b_s34b_b79k"  # 预训练权重

# 加载CLIP模型
def load_clip_model():
    print(f"加载CLIP模型: {Config.model_name}, 预训练权重: {Config.pretrained}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        Config.model_name, 
        pretrained=Config.pretrained,
        device=Config.device
    )
    print(f"模型加载完成，运行在: {Config.device}")
    return model, preprocess

# 提取图像特征向量
def extract_image_features(image_path, model, preprocess):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(Config.device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # 归一化特征向量
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy().squeeze()
    
    except Exception as e:
        print(f"处理图像 {os.path.basename(image_path)} 时出错: {str(e)}")
        return None

# 构建图像特征库
def build_feature_database():
    model, preprocess = load_clip_model()
    
    # 收集有效图像文件
    valid_images = []
    for img_name in os.listdir(Config.image_dir):
        if img_name.startswith('._') or not img_name.lower().endswith(('.jpg', '.jpeg', '.png','webp')):
            continue
        img_path = os.path.join(Config.image_dir, img_name)
        if os.path.isfile(img_path):
            valid_images.append(img_path)
    
    print(f"找到 {len(valid_images)} 张有效图像，开始提取特征...")
    
    # 提取所有图像特征
    features = []
    image_paths = []
    
    for img_path in tqdm(valid_images, desc="提取特征"):
        feature_vector = extract_image_features(img_path, model, preprocess)
        if feature_vector is not None:
            features.append(feature_vector)
            image_paths.append(img_path)
    
    # 保存特征向量和路径
    features_array = np.array(features)
    np.save(Config.feature_db, features_array)
    
    with open(Config.path_db, 'w') as f:
        f.write('\n'.join(image_paths))
    
    print(f"特征库已保存至 {Config.feature_db} 和 {Config.path_db}")
    
    return features_array, image_paths

# 加载特征库
def load_feature_database():
    if not os.path.exists(Config.feature_db) or not os.path.exists(Config.path_db):
        return build_feature_database()
    
    print(f"从 {Config.feature_db} 加载特征库...")
    features = np.load(Config.feature_db)
    
    with open(Config.path_db, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    print(f"成功加载 {len(image_paths)} 张图像的特征")
    return features, image_paths

# 检索相似图像
def retrieve_similar_images(query_image_path, features, image_paths, top_k=5):
    model, preprocess = load_clip_model()
    
    # 提取查询图像特征
    print(f"提取查询图像特征: {os.path.basename(query_image_path)}")
    query_feature = extract_image_features(query_image_path, model, preprocess)
    if query_feature is None:
        print("无法提取查询图像特征")
        return []
    
    # 计算余弦相似度
    similarities = cosine_similarity([query_feature], features)[0]
    
    # 获取最相似图像的索引
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 收集结果
    results = []
    for idx in top_indices:
        results.append({
            'image_path': image_paths[idx],
            'similarity': similarities[idx]
        })
    
    return results

# 可视化检索结果
def visualize_results(query_image_path, results):
    num_results = len(results)
    if num_results == 0:
        print("没有检索到结果")
        return
    
    # 创建图像网格
    fig, axes = plt.subplots(1, num_results + 1, figsize=(15, 4))
    
    # 显示查询图像
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title("查询图像", fontsize=10)
    axes[0].axis('off')
    
    # 显示结果图像
    for i, result in enumerate(results):
        result_img = Image.open(result['image_path'])
        axes[i+1].imshow(result_img)
        axes[i+1].set_title(f"相似度: {result['similarity']:.3f}", fontsize=10)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Top-{num_results} 相似图像检索结果", fontsize=12)
    plt.savefig("retrieval_results.jpg", dpi=150)
    plt.show()

# 主函数
def main():
    # 加载或构建特征库
    features, image_paths = load_feature_database()
    
    if len(image_paths) == 0:
        print("没有找到有效图像！")
        return
    
    # 随机选择一个查询图像
    query_idx = np.random.randint(0, len(image_paths))
    query_image_path = image_paths[query_idx]
    print(f"\n查询图像: {os.path.basename(query_image_path)}")
    
    # 检索相似图像
    results = retrieve_similar_images(query_image_path, features, image_paths, Config.top_k)
    
    if not results:
        print("没有检索到结果")
        return
    
    # 打印结果
    print("\n检索结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {os.path.basename(result['image_path'])} (相似度: {result['similarity']:.4f})")
    
    # 可视化结果
    visualize_results(query_image_path, results)

# 自定义查询
def custom_query(query_image_path):
    # 加载特征库
    features, image_paths = load_feature_database()
    
    if len(image_paths) == 0:
        print("没有找到有效图像！")
        return
    
    # 检查查询图像是否存在
    if not os.path.exists(query_image_path):
        print(f"查询图像不存在: {query_image_path}")
        return
    
    print(f"\n查询图像: {os.path.basename(query_image_path)}")
    
    # 检索相似图像
    results = retrieve_similar_images(query_image_path, features, image_paths, Config.top_k)
    
    if not results:
        print("没有检索到结果")
        return
    
    # 打印结果
    print("\n检索结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {os.path.basename(result['image_path'])} (相似度: {result['similarity']:.4f})")
    
    # 可视化结果
    visualize_results(query_image_path, results)

if __name__ == "__main__":
    # 使用随机查询图像运行
    #main()
    
    # 或者使用自定义查询图像
    custom_query("images/cry.webp")