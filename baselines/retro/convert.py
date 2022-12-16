with open('text_folder/base_data_128.txt') as f:
    dataset = ['\t'.join(line.strip().split('\t')[:-1]) for line in f.readlines()]
with open('text_folder/train.txt', 'w') as f:
    for line in dataset:
        line = line.replace('<|endoftext|>', '[UNK]')
        f.write(line + '\n')
