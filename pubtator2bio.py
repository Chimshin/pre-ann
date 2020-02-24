#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer as twt
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)


def pub2bio(text_clean, procedure, id_list, out_file, name):
    """
    change pubtator formatted data into bio format and save it into out_file

    text_clean: df with text data(title and abstract) (['id', 'full_text'])
    procedure: df with tag data (['id', 'start','end','word','tag','other'])
    id_list: list list of id for this func
    out_file: output path
    b_tag: str B-entity tag
    i-tag: str I-entity tag
    return: return 'done' when it's done
    """

    for i in id_list:
    #         with open(out_file, 'a') as f:
    #                 f.write('-DOCSTART- ('+i+')'+'\n'+'\n')

        # get full text of paper (title and abstract)
        whole = '. '.join(text_clean[text_clean.id == i]['full_text'])
        # tokenize the paragraph into sentences
        piece = nltk.sent_tokenize(whole)

        # get the char number of each sentence to match the span from pubtator
        lenth = [len(x)+1 for x in piece]
        lenth[0] -= 1 # we added a period for title(first sent)
        lenth =[0] + list(np.cumsum(lenth)) # sent len index

        # get the procedure tag and spans for paper i
        span_tag = procedure[procedure.id == i][['start','end','tag']]

        # for each sentence, get span match and change tag
        for j in range(len(piece)):
        # j = 12
            s = piece[j] # sentence j
            span_start, span_end = zip(*list(twt().span_tokenize(s))) # span for tokens in sentence j
            # df
            tmp = pd.DataFrame({'word': list(twt().tokenize(s)),
                                'w_start':span_start,
                                'w_end':span_end})
            # add char number from previous sentence
            tmp.loc[:,'w_start'] += lenth[j]
            tmp.loc[:,'w_end'] += lenth[j]
            # match with procedure tag
            tmp = tmp.merge(span_tag, how='left', left_on = 'w_start',
                            right_on = 'start')
            # change tag into bio format
            for k in tmp[tmp.start.notnull()].index:
                tmp.at[k,'tag'] = 'B-' + name
                end = tmp.at[k,'end']
                while tmp.at[k,'w_end'] < end :
                    k += 1
                    tmp.at[k,'tag'] = 'I-' + name
            tmp = tmp[['word', 'tag']].fillna('O')
            # output into outfile
            tmp.to_csv(out_file, header = None, index = None,
                       sep = '\t', mode = 'a')
            # add a new line after each sentence
            with open(out_file, 'a') as f:
                f.write('\n')

    return 'done'


def pre_pub(tag_list, file):
    """
    preprocess the pubtator ann file to get the tag data, text data and id list for next step

    tag_list: the target tag list to filter
    file: input pubtator formatted ann txt file

    return:
        tag_data: df with tag data (['id', 'start','end','word','tag','other'])
        pmid: list list of id with at least one target tag
        text_clean: df with text data(title and abstract) (['id', 'full_text'])
    """
    data = pd.read_csv(file, sep='\t',header=None,
                   names = ['id', 'start','end','word','tag','other'])
    tag_data = data[data.tag.isin(tag_list)]# get data with tags in tag list
    pmid = list(tag_data.id.unique())# get all the ids of tag data
    text = data[data.start.isnull()] # get the text without tag
    text_clean = pd.DataFrame({'id':text.id.apply(lambda x: x[:8]).values,
             'full_text':text.id.apply(lambda x: x[11:]).values }) # split the id and actual text
#     data.dropna(how = 'any', inplace = True)# drop the text, left all the tags

    return tag_data, pmid, text_clean


def train_test_dev_split(pmid, p_train = 0.7, p_test = 0.15, r1 = 42, r2 = 100):
    """
    split the id list into trian test dev list

    pmid: list list of id for this func
    p_train: train set per
    p_test: test set per
    r1: random state for first split
    r2: random state for second split

    return:
        train_pmid: list for train
        test_pmid: list for test
        dev_pmid: list for dev
    """
    train_pmid, other = train_test_split(pmid, test_size = (1 - p_train), random_state = r1)
    test_pmid, dev_pmid = train_test_split(other, test_size = p_test/(1 - p_train), random_state = r2)
    return train_pmid, test_pmid, dev_pmid


def get_name(out_dir, name):
    """
    name the out put train test dev txt file

    out_dir: str output folder path
    name: str name key word for this dataset

    return: list list of output path for train test dev txt file
    """
    return [str(Path(out_dir, 'bio.{}.train.txt'.format(name))),
            str(Path(out_dir, 'bio.{}.test.txt'.format(name))),
            str(Path(out_dir, 'bio.{}.dev.txt'.format(name)))]


if __name__ == '__main__':
    # get the procedure data from MedMention
    DATADIR = r'/Users/xuqinxin/nlp_data/MedMention'
    filename = 'corpus_pubtator.txt'

    name = 'procedure'
    OUTDIR = DATADIR+'/bio_' + name
    procedure_tag = ['T060']

    tag_data, pmid, text_clean = pre_pub(procedure_tag, Path(DATADIR, filename))

    train_pmid, test_pmid, dev_pmid = train_test_dev_split(pmid)
    file_id = [train_pmid, test_pmid, dev_pmid]
    out_name = get_name(OUTDIR, name)

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    for i in range(len(file_id)):
        pub2bio(text_clean, tag_data, file_id[i], out_name[i], name)
