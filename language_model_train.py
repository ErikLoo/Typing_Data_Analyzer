import nltk
from nltk.util import ngrams
from nltk.corpus import brown
from nltk import word_tokenize, sent_tokenize
from nltk.lm import MLE
from nltk.lm.models import KneserNeyInterpolated as KN
import pickle

brown_list = brown.words(categories='news')
# brown_list = brown.words()

wrd_list = [w for w in brown_list]
print("len of wrd list for training: " + str(len(wrd_list)))

# print(list(my_ngrams))
tokenized_text = [list(map(str.lower, list(wrd))) for wrd in wrd_list]

print(tokenized_text)
# Preprocess the tokenized text for 3-grams language modelling
from nltk.lm.preprocessing import padded_everygram_pipeline

n = 5
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

model = KN(n)
model.fit(train_data, padded_sents)

# save the model to file
f = open('KNLM.pickle', 'wb')
pickle.dump(model, f)
f.close()

print("LM generated and saved to file")
