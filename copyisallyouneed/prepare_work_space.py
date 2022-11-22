from config import load_base_config
import os, sys, ipdb

if __name__ == "__main__":
    config = load_base_config()

    datasets = config['datasets']
    models = config['models']
    root_dir = config['root_dir']

    for folder in ['rest', 'ckpt', 'log', 'bak']:
        path = f'{root_dir}/{folder}'
        if not os.path.exists(path):
            os.mkdir(path)
        for dataset in datasets:
            path = f'{root_dir}/{folder}/{dataset}'
            if not os.path.exists(path):
                os.mkdir(path)
            for model in models:
                path = f'{root_dir}/{folder}/{dataset}/{model}'
                if not os.path.exists(path):
                    os.mkdir(path)
        if folder == 'ckpt':
            for dataset in datasets:
                path = f'{root_dir}/ckpt/{dataset}/bert-post'
                if not os.path.exists(path):
                    os.mkdir(path)
        elif folder == 'log':
            # pipeline and pipeline_evaluation
            for dataset in datasets:
                path = f'{root_dir}/log/{dataset}/pipeline'
                if not os.path.exists(path):
                    os.mkdir(path)
    print(f'[!] init the folder under the {root_dir} over')
