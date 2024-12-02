import os
import pickle


# inp_name='ML-GCN/data/voc/voc_glove_word2vec.pkl'
# with open(inp_name, 'rb') as f:
#             inp = pickle.load(f)
#             print(inp.shape)
#         # self.inp_name = inp_name


# inp_name='ML-GCN/artifact_glove_word2vec.pkl'
# with open(inp_name, 'rb') as f:
#             inp = pickle.load(f)
#             print(inp.shape)
#         # self.inp_name = inp_name



inp_name='binary_matrix_0.5.pkl'
with open(inp_name, 'rb') as f:
            inp = pickle.load(f)
            print(type(inp))
        # self.inp_name = inp_name