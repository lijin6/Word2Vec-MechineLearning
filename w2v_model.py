from optparse import OptionParser
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence  # 修正1：正确导入LineSentence
import logging
import multiprocessing
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class W2v(object):
    def __init__(self, fin=None):
        self.model = None
        self.file = fin

    def load(self, fmodel):
        """加载预训练模型"""
        try:
            self.model = Word2Vec.load(fmodel)
            logging.info(f"成功加载模型: {fmodel}")
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            exit(1)

    def process(self, vector_size=100, window=5, min_count=5):  # 修正2：使用vector_size
        """训练Word2Vec模型"""
        if not os.path.exists(self.file):
            logging.error(f"输入文件不存在: {self.file}")
            exit(1)
            
        try:
            workers = multiprocessing.cpu_count()
            sentences = LineSentence(self.file)  # 修正3：直接使用LineSentence
            
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,  # Gensim 4.0+ 参数名
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=10  # 新版参数
            )
            logging.info(f"训练完成，词表大小: {len(self.model.wv)}")
            
        except Exception as e:
            logging.error(f"训练过程中出错: {str(e)}")
            exit(1)

    def save(self, fout, binary=False):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(fout), exist_ok=True)
            
            if binary:
                self.model.wv.save_word2vec_format(fout, binary=True)
                logging.info(f"模型已保存为二进制格式: {fout}")
            else:
                self.model.save(fout)
                logging.info(f"模型已保存: {fout}")
                
        except Exception as e:
            logging.error(f"保存失败: {str(e)}")
            exit(1)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--file', type=str, help='已分词的语料库文件（每行一个句子，空格分隔）', dest='wordsfile')
    parser.add_option('-m', '--model', type=str, help='模型保存路径', dest='modelfile')
    parser.add_option('-s', '--size', type=int, default=100, help='词向量维度（默认100）', dest='size')
    parser.add_option('-b', '--binary', action='store_true', default=False, help='保存为二进制格式', dest='binary')

    options, args = parser.parse_args()
    
    if not options.wordsfile:
        parser.print_help()
        logging.error("必须指定输入文件！")
        exit(1)

    w = W2v(options.wordsfile)
    logging.info('开始训练...')
    w.process(vector_size=options.size)  # 注意使用vector_size
    logging.info('训练完成')
    w.save(options.modelfile, binary=options.binary)