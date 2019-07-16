from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
import keras.backend as K

if len(K.tensorflow_backend._get_available_gpus()) > 0:
	from keras.layers import CuDNNLSTM as LSTM
	from keras.layers import CuDNNGRU as GRU