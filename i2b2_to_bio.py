from pubtator2bio import train_test_dev_split, get_name

from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 500)


def ib2bio(file, text, tag_line, df_tag, name):
    """
    change i2b2 formatted data into bio format and save it into file

    file: Path obj to output
    text: list of sentences in one medical record
    tag_line: list of line number in medical record with tag
    df_tag: dataframe of tag information (line#, token#, length, tag)
    name: str of BIO tag name

    return: 0
    """
    with file.open('a+') as g:
        # for each line in text
        for i in range(len(text)):
            this_line = i + 1 # match line # in annotation
            this_text = text[i].rstrip('\n').split()  # all tokens in this line
            if this_line in tag_line: # if this line has tag
                this_tag = df_tag[df_tag.line == this_line] # all the tags in this line

                # exam each token and add tag
                j = 0 # counter
                while j < len(this_text):
                    if j in this_tag.token.values: # check if this token is the start of any tagged entity
                        tmp = this_tag[this_tag.token == j] # all info about this entity
                        bound = j + tmp.length.values[0] # get the end of this entity
                        # g.write(this_text[j] + '\t' + 'B-' + str(tmp.tag.values[0]) + '\n')
                        g.write(this_text[j] + '\t' + 'B-' + name + '\n') # b tag for first token
                        j += 1
                        while j < bound:  # i tag for the following tokens
                            # g.write(this_text[j] + '\t' + 'I-' + str(tmp.tag.values[0]) + '\n')
                            g.write(this_text[j] + '\t' + 'I-' + name + '\n')
                            j += 1
                    else: # if not the start of any tagged entity
                        g.write(this_text[j]+'\t'+'O'+'\n')
                        j += 1
            else: # if no tag in this line
                for j in range(len(this_text)):
                    g.write(this_text[j]+'\t'+'O'+'\n')
            g.write('\n') # split line
    return 0


def pre_ib(DATADIR, file, tag_list):
    """
    preprocess the i2b2 ann file and original txt file to get the df_tag data, text data and tag line list for next step

    DATADIR: str of the main path for dataset
    file: str of file name
    tag_list: list of tag you want to transform

    return:
        text: list of sentences in one medical record
        df_tag: dataframe of tag information (line#, token#, length, tag)
        tag_line: list of line number in medical record with tag
    """

    # read in the original txt file
    with Path(DATADIR+'/i2b2-2010-all-txt/' + file + '.txt').open('r') as f:
        text = f.readlines()
    # read in the entity ann file
    with Path(DATADIR+'/beth/concept/' + file + '.con').open('r') as f:
        tag = []
        for tmp in f.readlines():
            tmp = tmp.split('"')
            if tmp[3] in tag_list:
                line = []  # line# token# length tag
                line.extend(tmp[2].split()[0].split(':'))
                line.append(len(tmp[1].split()))# length of tagged tokens
                line.append(tmp[3])
                tag.append(line)

    # put into df
    df_tag = pd.DataFrame(tag, columns=['line','token', 'length', 'tag'])\
    # change data type
    df_tag = df_tag.astype(dtype={"line": "int16",
                                  "token": "int16",
                                  "length": "int16",
                                  'tag': 'str'})
    # get the list of line #
    tag_line = list(df_tag.line.unique())

    return text, df_tag, tag_line


def output4list(DATADIR, file_name, tag_list, name):
    """
    create output folders/files and do the transfom

    DATADIR: str of the main path for dataset
    file_name: list of list of file names ([train, test, dev])
    tag_list: list of tag you want to transform
    name: str of BIO tag name and output folder name and output file name
    return: 'done'
    """

    OUTDIR = DATADIR + '/bio_' + name

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    out_name = get_name(OUTDIR, name)

    # for split in file_name:
    for i in range(len(file_name)):
        for file in file_name[i]:
            text, df_tag, tag_line = pre_ib(DATADIR, file, tag_list)
            ib2bio(out_name[i], text, tag_line, df_tag, name)

    return 'done'


if __name__ == '__main__':
    DATADIR = '/Users/xuqinxin/nlp_data/concept_assertion_relation_data/'

    file_name = [str(item.stem) for item in Path(DATADIR+'/beth/concept/').iterdir() if item.suffix == '.con']

    train_name, test_name, dev_name = train_test_dev_split(file_name, r1=13, r2=128)
    file_name = [train_name, test_name, dev_name]

    tag_list = ['problem']
    name = 'disease'
    output4list(DATADIR, file_name, tag_list, name)

    tag_list = ['test', 'treatment']
    name = 'procedure'
    output4list(DATADIR, file_name, tag_list, name)

    print('finished')