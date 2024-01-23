from logging import raiseExceptions
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import re
import torch
from sentence_transformers import SentenceTransformer


def preprocess_data(df, mode):
    x_data, y_data = [], []
    if len(df) % 20 != 0:
        print('error length')
        return

    num_windows = int(len(df) / 20)
    for i in tqdm(range(num_windows)):
        df_blk = df[i*20:i*20+20]
        x_data.append(np.array(df_blk["Vector"].tolist()))
        labels = df_blk["Label"].tolist()
        if labels == ['-']*20:
            y = [1, 0]
        else:
            y = [0, 1]
        y_data.append(y)

    np.savez(f'preprocessed_data/{log_name}_{mode}_data.npz',
             x=x_data, y=y_data)


if __name__ == '__main__':
    num_workers = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        'distilbert-base-nli-mean-tokens', device=device)

    # load data
    log_name = 'BGL'
    df_template = pd.read_csv(f"parse_result/{log_name}.log_templates.csv")
    df_structured = pd.read_csv(f"parse_result/{log_name}.log_structured.csv")

    # calculate vectors for all known templates
    print('vector embedding...')
    embeddings = model.encode(
        df_template['EventTemplate'].tolist())  # num_workers=num_workers)
    df_template['Vector'] = list(embeddings)
    template_dict = df_template.set_index('EventTemplate')['Vector'].to_dict()

    # convert templates to vectors for all logs
    vectors = []
    for idx, template in enumerate(df_structured['EventTemplate']):
        try:
            vectors.append(template_dict[template])
        except KeyError:
            # new template
            vectors.append(model.encode(template))
    df_structured['Vector'] = vectors
    print('done')
    df_structured.drop(
        columns=['Date', 'Node', 'Time', 'NodeRepeat', 'Type', 'Component', 'Level'])

    num_windows = len(df_structured)//20
    df_structured = df_structured.iloc[:num_windows*20]

    training_windows = (num_windows//5)*4
    df_structured['Usage'] = 'testing'
    df_structured.iloc[:training_windows*20,
                       df_structured.columns.get_loc('Usage')] = 'training'

    df_test = df_structured[df_structured['Usage'] == 'testing']
    df_train = df_structured[df_structured['Usage'] == 'training']

    # preprocess data
    preprocess_data(df_train, 'training')
    preprocess_data(df_test, 'testing')
