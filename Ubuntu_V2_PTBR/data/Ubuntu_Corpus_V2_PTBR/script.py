# %%
import pandas as pd


# %%
dataset = pd.read_csv("test.csv")
dataset2 = pd.read_csv("train.csv")
dataset3 = pd.read_csv("valid.csv")


# %%
def get_char_vocab(df, vocab_char):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    for column in df.columns:
        if column != "Label":
            for words in df[column]:
                for word in words:
                    vocab_char.update(word)

    return vocab_char

def get_vocab(df , vocab):
    for column in df.columns:
        if column != "Label":
            vocab.update(set(dataset2[column].str.cat(sep=' ').split()))

    return vocab

# %%
vocab = {"<PAD>", "UNKNOWN"}

vocab = get_vocab(dataset2,vocab)

with open("vocab.txt",'w') as f2:
    for index, item in enumerate(vocab):
        f2.write("{}	{}\n".format( item, index))

vocab_char = set()

char_vocab = get_char_vocab(dataset2, vocab_char)
char_vocab = get_char_vocab(dataset, vocab_char)
char_vocab = get_char_vocab(dataset3, vocab_char)

with open("char_vocab.txt",'w') as f:
    for index, item in enumerate(sorted(char_vocab)):
        f.write("{}	{}\n".format(index, item))


# %%
def response_train (df,response):

    for index, row in df.iterrows():
        if row['Label'] == 1.0:
            if row['Utterance'] in response:
                df.loc[index, 'Utterance'] = response.index(row['Utterance'])
                # row['Utterance'] = response.index(row['Utterance'])
                df.loc[index,'Label'] = "NA"
            else:
                response.append(row['Utterance'])
                df.loc[index, 'Utterance'] = response.index(row['Utterance'])
                df.loc[index,'Label'] = "NA"
        else:
            if row['Utterance'] in response:
                df.loc[index,'Label'] = response.index(row['Utterance'])
                df.loc[index, 'Utterance'] =  "NA"
                
            else:
                response.append(row['Utterance'])
                df.loc[index,'Label'] = response.index(row['Utterance'])
                df.loc[index, 'Utterance'] = "NA"
                
    return df, response

def response_test_valid (df, response):
    for index, row in df.iterrows():
        if row['Ground Truth Utterance'] in response:
            df.loc[index,'Ground Truth Utterance'] = response.index(row['Ground Truth Utterance'])
        else:
            response.append(row['Ground Truth Utterance'])
            df.loc[index,'Ground Truth Utterance'] = response.index(row['Ground Truth Utterance'])
        
        for i in range(9):
            if row['Distractor_{}'.format(i)] in response:
               df.loc[index,'Distractor_{}'.format(i)] = response.index(row['Distractor_{}'.format(i)])
            else:
                response.append(row['Distractor_{}'.format(i)])
                df.loc[index,'Distractor_{}'.format(i)] = response.index(row['Distractor_{}'.format(i)])
    return df,response

def format_test_valid(df):
    distractors_list = []
    for index, row in df.iterrows():
        distractors = []
        for i in range(9):
            distractors.append(row['Distractor_{}'.format(i)])
        distractors = '|'.join(str(v) for v in distractors)
        distractors_list.append(distractors)
    df['Distractors'] = distractors_list

    df.drop(columns=['Distractor_0','Distractor_1','Distractor_2','Distractor_3','Distractor_4','Distractor_5','Distractor_6','Distractor_7','Distractor_8'], axis=1, inplace = True)
    return df

# %%
response = []
dataset2, response = response_train(dataset2, response)

# with open('train.txt', "w") as f2:
#     [f2.write("	".join(row)+'\n') for row in dataset2]
dataset2['Context'] = dataset2['Context'].str.replace('|',' ',regex=True)
dataset2['Context'] = dataset2['Context'].str.replace('\t',' ',regex=True)


dataset2.to_csv('train.txt', sep='\t', header=None)

dataset, response = response_test_valid(dataset, response)
dataset = format_test_valid(dataset)
dataset['Context'] = dataset['Context'].str.replace('\t',' ',regex=True)
# with open('test.txt', "w") as f1:
#     [f1.write("	".join(row)+'\n') for row in dataset]

dataset.to_csv('test.txt', sep='\t', header=None)

dataset3, response = response_test_valid(dataset3, response)
dataset3 = format_test_valid(dataset3)
dataset3['Context'] = dataset3['Context'].str.replace('\t',' ',regex=True)
# with open('valid.txt', "w") as f3:
#     [f3.write("	".join(row)+'\n') for row in dataset3]

dataset3.to_csv('valid.txt', sep='\t', header=None)

responsefixed = [item.replace('\t', ' ') for item in response]

with open("responses.txt",'w') as f:
    for index, item in enumerate(responsefixed):
        f.write("{}	{}\n".format(index, item))

# response.to_csv('responses.txt', sep='\t', header=None)



