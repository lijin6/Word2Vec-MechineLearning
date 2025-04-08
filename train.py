from optparse import OptionParser
import pandas as pd
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import warnings
from gensim.models import Word2Vec

# 配置日志和忽略警告
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def evaluate_model(model, X, y, cv=10):
    """执行交叉验证并返回多指标结果"""
    scoring = {
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }
    
    cv_results = sk_model_selection.cross_validate(
        model, X, y, 
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )
    
    return {
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall'],
        'f1': cv_results['test_f1']
    }

def plot_results(results, save_dir="images"):
    """生成三种可视化图表并保存"""
    os.makedirs(save_dir, exist_ok=True)
    palette = {
        "GaussianNB": "#E74C3C",  # 红色
        "LogisticRegression": "#3498DB",  # 蓝色
        "LinearSVC": "#9B59B6",  # 紫色
        "RandomForest": "#2ECC71",  # 绿色
        "XGBoost": "#F39C12"  # 橙色
    }
    
    # 准备数据
    metrics_df = pd.DataFrame({
        model: [
            np.mean(results[model]['precision']),
            np.mean(results[model]['recall']),
            np.mean(results[model]['f1'])
        ] 
        for model in results
    }, index=['Precision', 'Recall', 'F1']).T
    
    # ============= 1. 综合指标柱状图 =============
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=metrics_df.reset_index().melt(id_vars="index"), 
                    x="index", y="value", hue="variable",
                    palette=["#3498DB", "#E74C3C", "#2ECC71"])
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   fontsize=10, color='black',
                   xytext=(0, 5),
                   textcoords='offset points')
    
    plt.title("Model Performance Comparison (Mean Scores)", fontsize=14, pad=20)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "1_metrics_comparison.png"), 
               bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    # ============= 2. 精确率折线图 =============
    plt.figure(figsize=(14, 7))
    for model in results:
        plt.plot(results[model]['precision'], 
                label=model, 
                color=palette[model], 
                marker='o', 
                linewidth=2.5,
                markersize=8)
    
    plt.title("Precision Across 10-Fold Cross Validation", fontsize=14, pad=20)
    plt.xlabel("Fold Number", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xticks(range(10), [f"Fold {i+1}" for i in range(10)])
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(os.path.join(save_dir, "2_precision_details.png"), 
               bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    # ============= 3. 指标分布箱线图 =============
    plt.figure(figsize=(14, 6))
    data_for_boxplot = []
    for model in results:
        for metric in ['precision', 'recall', 'f1']:
            for value in results[model][metric]:
                data_for_boxplot.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Value': value
                })
    
    df_box = pd.DataFrame(data_for_boxplot)
    
    sns.boxplot(data=df_box, x='Model', y='Value', hue='Metric',
               palette={"Precision": "#3498DB", "Recall": "#E74C3C", "F1": "#2ECC71"},
               width=0.7)
    
    plt.title("Distribution of Evaluation Metrics", fontsize=14, pad=20)
    plt.xlabel("")
    plt.ylabel("Score Value", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(os.path.join(save_dir, "3_metrics_distribution.png"), 
               bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def main():
    # 参数解析
    parser = OptionParser()
    parser.add_option('-i', '--in', type=str, help='语料库文件路径', dest='corpusfile')
    parser.add_option('-l', '--label', type=str, help='标签列名', dest='label')
    parser.add_option('-d', '--data', type=str, help='文本数据列名', dest='data')
    parser.add_option('-m', '--model', type=str, help='Word2Vec模型路径', dest='model')
    
    options, _ = parser.parse_args()
    
    # 参数默认值
    params = {
        'corpusfile': './data/2/words_type.csv',
        'model': './data/2/w2v.bin',
        'label': 'type',
        'data': 'words'
    }
    for k, v in params.items():
        if getattr(options, k, None):
            params[k] = getattr(options, k)
    
    # 数据加载
    try:
        df = pd.read_csv(params['corpusfile'], sep='\t')
        logging.info(f"成功加载数据: {params['corpusfile']}")
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        exit(1)
    
    # 加载Word2Vec模型
    try:
        w2v_model = Word2Vec.load(params['model'])
        logging.info(f"成功加载Word2Vec模型: {params['model']}")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        exit(1)
    
    # 数据验证
    for col in [params['label'], params['data']]:
        if col not in df.columns:
            logging.error(f"列不存在: {col}")
            exit(1)
    
    # 生成词向量特征
    X = []
    for text in df[params['data']]:
        vectors = [w2v_model.wv[word] for word in str(text).split() if word in w2v_model.wv]
        if vectors:
            X.append(np.mean(vectors, axis=0))
        else:
            X.append(np.nan)
    
    df['w2v'] = X
    df = df.dropna()
    y = df[params['label']].values
    X = np.vstack(df['w2v'].values)
    
    logging.info(f"最终数据集: {len(X)}条样本 (正样本: {sum(y==1)}, 负样本: {sum(y==-1)})")
    
    # 定义模型集合
    models = {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5),
      
    }
    
    # 评估所有模型
    results = {}
    for name, model in models.items():
        logging.info(f"开始评估模型: {name}")
        results[name] = evaluate_model(model, X, y)
        
        # 打印结果
        logging.info(f"{name} 评估结果:")
        for metric in ['precision', 'recall', 'f1']:
            mean_val = np.mean(results[name][metric])
            std_val = np.std(results[name][metric])
            logging.info(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 可视化结果
    plot_results(results)

if __name__ == '__main__':
    main()