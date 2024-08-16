#####Welcome!

from gensim.models import KeyedVectors
import tensorflow as tf
import pandas as pd
complete_set = pd.read_csv('complete_set_1.csv')
protein_vector_dict = pd.read_csv('ROBOMechDB Protein Vector Dictionary.csv')

triples_drug_keys = complete_set['0']
triples_disease_keys = complete_set['1']
triples_protein_keys = complete_set['2']

triples_dictionary = {}
values = my_list = [1] * len(triples_drug_keys)
keys = [triples_drug_keys[i] + " " + triples_disease_keys[i]+ " " + triples_protein_keys[i] for i in range(0,len(triples_drug_keys))]
for key,value in zip(keys,values):
    triples_dictionary[key] = value

##make sure you install the correct version of scipy (v1.12 or earlier or else this block wont work)

biowordvec_model = KeyedVectors.load_word2vec_format('BioWordVec Embedding Model.bin', binary=True)

def get_phrase_vec(vocablist, phrase):
    words = phrase.replace(",", "")
    words_split = words.split(' ')
    # print(words)
    count = 0
    flag = 0
    for i in words_split:
        if i in vocablist:
            if count == 0:
                comb_emb = biowordvec_model[i]
            else:
                comb_emb = biowordvec_model[i] + comb_emb
            count = count + 1
        else:
            flag = 1 + flag
            break
    if flag != 0:
        return flag, phrase, "no embedding"
        # print(phrase, "no embedding")
        flag = flag
    if flag == 0:
        return flag, phrase, list(comb_emb)


vocab = biowordvec_model.index_to_key

robomechdb_model = tf.keras.models.load_model('ROBOMechDB Complete Set 1 Model Fold 1.keras')


print('Welcome to our ROBOMechDB Classification Model Prediction Interface! All you have to do is enter a drug, disease, and protein, and our model will make an informed prediction on how likely it is to be a therapeutic combination!')

def get_valid_input(prompt, vocab):
    while True:
        user_input = input(prompt)
        fl, ph, st = get_phrase_vec(vocab, user_input)
        if st == 'no embedding':
            print(f"Sorry, our model can't encode your {prompt.split()[-2].lower()}.")
        else:
            return fl, ph, st

    # Get valid drug, disease, and protein inputs
fl_drug, ph_drug, str_drug = get_valid_input('Please enter a drug name: ', vocab)
fl_disease, ph_disease, str_disease = get_valid_input('Please enter a disease name: ', vocab)

    # Check if the combination is already in the model database
print('Now, our model will recommend you the most likely proteins that the drug can bind to to treat the disease!')

import numpy as np
import itertools

combinations_array = []

for i in range(len(protein_vector_dict)):
    protein_name = protein_vector_dict.iloc[i,0]
    if (ph_drug + " " + ph_disease + " " + protein_name) in triples_dictionary:
        continue
    protein_vector = protein_vector_dict.iloc[i,1:]
    row = [[ph_drug], [ph_disease], [protein_name], str_drug, str_disease, protein_vector]
    merged = list(itertools.chain(*row))
    combinations_array.append(merged)

combinations_array = np.array(combinations_array)
combinations_df = pd.DataFrame(combinations_array)

input_data = np.array(combinations_df.iloc[:,3:])
input_data = np.array([[float(x) for x in y] for y in input_data])

predictions = robomechdb_model.predict(input_data)
descending_confidence_values = np.sort(predictions.flatten())[::-1]
ascending_indices = np.argsort(predictions.flatten())
descending_indices = np.flip(ascending_indices)

prediction_array = []
for i in descending_indices:
    triple_name = combinations_df.iloc[i,:3].tolist()
    conf_value = float(predictions[i])
    row = [triple_name,conf_value]
    flat_row = row[0]+[row[1]]
    prediction_array.append(flat_row)

columns_to_use = ['drug_name','disease_name','protein_name','model_confidence_value']
df = pd.DataFrame(data = prediction_array,columns = columns_to_use)
print(df)




