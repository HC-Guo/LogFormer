import ast
import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import Drain

log_name = 'HDFS_2k'
input_dir = 'log_data/'  # The input directory of log file
output_dir = 'parse_result/'  # The output directory of parsing results


def preprocess_data(df, mode):
    x_data, y_data = [], []
    pbar = tqdm(total=df['BlockId'].nunique(),
                desc=f'{mode} data collection')

    while len(df) > 0:
        blk_id = df.iloc[0]['BlockId']
        last_index = 0
        for i in range(len(df)):
            if df.iloc[i]['BlockId'] != blk_id:
                break
            last_index += 1

        df_blk = df[:last_index]
        x_data.append(np.array(df_blk['Vector'].tolist()))

        y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
        y = [0, 0]
        y[y_index] = 1
        y_data.append(y)

        df = df.iloc[last_index:]
        pbar.update()
    pbar.close()

    np.savez(f'preprocessed_data/{log_name}_{mode}.npz',
             x=x_data, y=y_data)


if __name__ == '__main__':
    if not os.path.exists(output_dir+log_name+'.log_structured.csv'):
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = Drain.LogParser(log_format, indir=input_dir,
                                 outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_name+'.log')

    num_workers = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        'distilbert-base-nli-mean-tokens', device=device)

    structured_file_name = log_name+'.log_structured.csv'
    template_file_name = log_name+'.log_templates.csv'

    # load data
    df_template = pd.read_csv(output_dir + template_file_name)
    df_structured = pd.read_csv(output_dir + structured_file_name)
    df_label = pd.read_csv(input_dir+'anomaly_label.csv')

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

    # remove unused column
    df_structured.drop(columns=['Date', 'Time', 'Pid', 'Level', 'Component',
                                'Content', 'EventId', 'EventTemplate'], axis=1, inplace=True)

    # extract BlockId
    r1 = re.compile('^blk_-?[0-9]')
    r2 = re.compile('.*blk_-?[0-9]')

    paramlists = df_structured['ParameterList'].tolist()
    blk_id_list = []
    for paramlist in tqdm(paramlists, desc='extract BlockId'):
        paramlist = ast.literal_eval(paramlist)
        blk_id = list(filter(r1.match, paramlist))

        if len(blk_id) == 0:
            filter_str_list = list(filter(r2.match, paramlist))
            # ex: '/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906'
            blk_id = filter_str_list[0].split(' ')[-1]
        else:
            # ex: ['blk_-1608999687919862906'], ['blk_-1608999687919862906', 'blk_-1608999687919862906'],
            # ['blk_-1608999687919862906 terminating']
            blk_id = blk_id[0].split(' ')[0]

        blk_id_list.append(blk_id)

    df_structured['BlockId'] = blk_id_list
    df_structured.drop(columns=['ParameterList'], axis=1, inplace=True)

    # split training and testing data labels
    df_label['Usage'] = 'testing'
    train_index = df_label.sample(frac=0.2, random_state=123).index
    df_label.iloc[train_index, df_label.columns.get_loc('Usage')] = 'training'

    # n_index = df_label.Label[df_label.Label.eq('Normal')].sample(6000).index
    # a_index = df_label.Label[df_label.Label.eq('Anomaly')].sample(6000).index
    # train_index = n_index.union(a_index)
    # df_label.iloc[train_index, df_label.columns.get_loc('Usage')] = 'training'

    df_structured = pd.merge(df_structured, df_label, on='BlockId')
    del df_label

    # group data by BlockId
    df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
    df_structured.drop(columns=['LineId'], axis=1, inplace=True)

    # split training and testing dataframe
    df_test = df_structured[df_structured['Usage'] == 'testing']
    df_train = df_structured[df_structured['Usage'] == 'training']
    del df_structured

    # preprocess data
    preprocess_data(df_train, 'training')
    preprocess_data(df_test, 'testing')
