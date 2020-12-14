import pickle
f = open('KNLM.pickle', 'rb')
model = pickle.load(f)
f.close()


# test
# print(model.score('d', list('plea')))  # P('is'|'language')
# print(model.score('s', list('plea')))
# print(model.score('t', list('plea')))
# print(model.score('l', list('plea')))  # P('is'|'language')