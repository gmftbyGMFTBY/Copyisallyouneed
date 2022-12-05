import faiss
import joblib

class Searcher:

    def __init__(self, index_type, dimension=768, nprobe=1):
        self.searcher = faiss.index_factory(dimension, index_type, faiss.METRIC_INNER_PRODUCT)
        self.corpus = []
        self.source_corpus = {}
        self.nprobe = nprobe
        self.index_type = index_type

    def _build(self, matrix, corpus, source_corpus=None, speedup=False):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        if speedup:
            self.move_to_gpu()
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        if speedup:
            self.move_to_cpu()
        print(f'[!] build collection with {self.searcher.ntotal} samples')
    
    def _search(self, vector, topk=20):
        self.searcher.nprobe = self.nprobe
        D, I = self.searcher.search(vector, topk)
        rest = [[self.corpus[i] for i in N] for N in I]
        distance = [[i for i in N] for N in D]
        return rest, distance

    def save(self, path_faiss, path_corpus, path_source_corpus=None):
        faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)

    def load(self, path_faiss, path_corpus, path_source_corpus=None):
        self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)
        print(f'[!] load {len(self.corpus)} utterances from {path_faiss} and {path_corpus}')

    def add(self, vectors, texts):
        '''the whole source information are added in _build'''
        self.searcher.add(vectors)
        self.corpus.extend(texts)
        print(f'[!] add {len(texts)} dataset over')

    def move_to_gpu(self, device=0):
        res = faiss.StandardGpuResources()
        self.searcher = faiss.index_cpu_to_gpu(res, device, self.searcher)
        print(f'[!] move index to GPU device: {device} over')
    
    def move_to_cpu(self):
        self.searcher = faiss.index_gpu_to_cpu(self.searcher)
        print(f'[!] move index from GPU to CPU over')
