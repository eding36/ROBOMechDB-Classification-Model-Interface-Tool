#####Welcome!

from gensim.models import KeyedVectors
import tensorflow as tf

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
import pandas as pd
complete_set = pd.read_csv('complete_set.csv')

triples_drug_keys = complete_set['0']
triples_disease_keys = complete_set['1']
triples_protein_keys = complete_set['2']

triples_dictionary = {}
values = my_list = [1] * len(triples_drug_keys)
keys = [triples_drug_keys[i] + " " + triples_disease_keys[i]+ " " + triples_protein_keys[i] for i in range(0,len(triples_drug_keys))]
for key,value in zip(keys,values):
    triples_dictionary[key] = value

print('Welcome to our ROBOMechDB Classification Model Prediction Interface! All you have to do is enter a drug, disease, and protein, and our model will make an informed prediction on how likely it is to be a therapeutic combination!')

def get_valid_input(prompt, vocab):
    while True:
        user_input = input(prompt)
        fl, ph, st = get_phrase_vec(vocab, user_input)
        if st == 'no embedding':
            print(f"Sorry, our model can't encode your {prompt.split()[-2].lower()}.")
        else:
            return fl, ph, st

while True:
    # Get valid drug, disease, and protein inputs
    fl_drug, ph_drug, str_drug = get_valid_input('Please enter a drug name: ', vocab)
    fl_disease, ph_disease, str_disease = get_valid_input('Please enter a disease name: ', vocab)
    fl_protein, ph_protein, str_protein = get_valid_input('Please enter a protein name: ', vocab)

    # Check if the combination is already in the model database
    if (ph_drug + " " + ph_disease + " " + ph_protein) in triples_dictionary:
        print('The combination you inputted is already in our model database. Please try a different combo!')
    else:
        break

import numpy as np
vector = str_drug + str_disease + str_protein
vector = np.array(vector).reshape(1,-1)
prediction = robomechdb_model.predict(vector).flatten()[0]

print(f'Our model is {round(prediction*100, 2)} percent confident that the combination you just entered is a therapeutic triple!')

