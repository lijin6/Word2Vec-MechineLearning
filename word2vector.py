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
import warnings
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def plot_performance(results, save_dir="images"):
    """绘制并保存模型性能对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 箱线图：展示精度分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, palette="Set2")
    plt.title("Model Precision Comparison (10-fold CV)")
    plt.ylabel("Precision Score")
    plt.xticks(rotation=45)
    boxplot_path = os.path.join(save_dir, "model_comparison_boxplot.png")
    plt.savefig(boxplot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"箱线图已保存至: {boxplot_path}")
    
    # 柱状图：展示均值精度
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df_results.columns, y=df_results.mean(), palette="viridis")
    plt.title("Average Precision by Model")
    plt.ylabel("Mean Precision")
    plt.xticks(rotation=45)
    for i, v in enumerate(df_results.mean()):
        plt.text(i, v+0.01, f"{v:.3f}", ha='center')
    barplot_path = os.path.join(save_dir, "model_comparison_barplot.png")
    plt.savefig(barplot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"柱状图已保存至: {barplot_path}")

if __name__ == '__main__':
    # 参数解析
    parser = OptionParser()
    parser.add_option('-i', '--in', type=str, help='语料库文件', dest='corpusfile')
    parser.add_option('-l', '--label', type=str, help='标签列名', dest='label')
    parser.add_option('-d', '--data', type=str, help='数据列名', dest='train')
    parser.add_option('-m', '--model', type=str, help='w2v模型文件名', dest='model')
    options, args = parser.parse_args()

    # 参数默认值
    parm_corpusfile = './data/1/words_type.csv' if not options.corpusfile else options.corpusfile
    parm_model = './data/1/w2v.bin' if not options.model else options.model
    parm_label = 'type' if not options.label else options.label
    parm_data = 'words' if not options.train else options.train

    # 日志记录参数
    logging.info(f'语料库文件: {parm_corpusfile}')
    logging.info(f'w2v模型文件: {parm_model}')
    logging.info(f'标签列名: {parm_label}')
    logging.info(f'数据列名: {parm_data}')

    # 数据加载
    try:
        df = pd.read_csv(parm_corpusfile, sep='\t')
    except Exception as e:
        logging.error(f'{parm_corpusfile} 语料库文件路径错误！{str(e)}')
        exit()

    # 模型加载
    try:
        w2vmodel = Word2Vec.load(parm_model)
    except Exception as e:
        logging.error(f'{parm_model} w2v模型加载失败！{str(e)}')
        exit()

    # 数据验证
    if parm_label not in df.columns:
        logging.error(f'{parm_label} 标签列不存在！')
        exit()
    if parm_data not in df.columns:
        logging.error(f'{parm_data} 数据列不存在！')
        exit()

    # 生成词向量特征
    ct = []
    for text in df[parm_data]:
        vectors = [w2vmodel.wv[word] for word in text.split() if word in w2vmodel.wv]
        ct.append(np.mean(vectors, axis=0) if vectors else np.nan)
    
    df['w2v'] = ct
    df = df.dropna()
    logging.info(f'有效数据量: {len(df)} (正样本: {len(df[df[parm_label]==1])}, 负样本: {len(df[df[parm_label]==-1])})')

    # 准备训练数据
    X = np.vstack(df['w2v'].values)
    y = df[parm_label].values

    # 模型训练与评估
    models = {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(),
        "LinearSVC": LinearSVC()
    }
    
    results = {}
    for name, model in models.items():
        accs = sk_model_selection.cross_val_score(
            model, X, y, 
            scoring='precision', cv=10, n_jobs=1
        )
        results[name] = accs
        print(f'{name} 交叉验证结果: {accs.round(3)}, 均值: {np.mean(accs):.3f}')

    # 可视化结果
    plot_performance(results)