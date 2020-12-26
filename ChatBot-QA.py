


import pickle
import numpy as np



with open("train_qa.txt", "rb") as fp:   # Unpickling
    train_data =  pickle.load(fp)



with open("test_qa.txt", "rb") as fp:   # Unpickling
    test_data =  pickle.load(fp)



type(test_data)
type(train_data)
len(test_data)
len(train_data)
train_data[0]

' '.join(train_data[0][0])


' '.join(train_data[0][1])

train_data[0][2]

vocab = set()

all_data = test_data + train_data

for story, question , answer in all_data:
    
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


vocab.add('no')
vocab.add('yes')

vocab

max_story_len = max([len(data[0]) for data in all_data])

max_story_len
max_question_len = max([len(data[1]) for data in all_data])
max_question_len

vocab

vocab_size = len(vocab) + 1





from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

tokenizer.word_index

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)


train_story_seq = tokenizer.texts_to_sequences(train_story_text)

len(train_story_text)

len(train_story_seq)





def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len):
    
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []
    
    
    for story, query, answer in data:
        
        
        x = [word_index[word.lower()] for word in story]
        
        xq = [word_index[word.lower()] for word in query]
        
        y = np.zeros(len(word_index) + 1)
        
        
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)

inputs_test, queries_test, answers_test = vectorize_stories(test_data)

inputs_test

queries_test

answers_test

sum(answers_test)

tokenizer.word_index['yes']

tokenizer.word_index['no']

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM


input_sequence = Input((max_story_len,))
question = Input((max_question_len,))



input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_question_len))
question_encoder.add(Dropout(0.3))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)



match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)



response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
answer = concatenate([response, question_encoded])
answer
answer = LSTM(32)(answer)  # (samples, 32)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)



answer = Activation('softmax')(answer)
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])





model.summary()



history = model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))

filename = 'chatbot_120_epochs.h5'
model.save(filename)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model.load_weights(filename)
pred_results = model.predict(([inputs_test, queries_test]))

test_data[0][0]

story =' '.join(word for word in test_data[0][0])
print(story)

query = ' '.join(word for word in test_data[0][1])
print(query)
print("True Test Answer from Data is:",test_data[0][2])






val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
vocab
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()
my_question = "Is the football in the garden ?"
my_question.split()
mydata = [(my_story.split(),my_question.split(),'yes')]


my_story,my_ques,my_ans = vectorize_stories(mydata)


pred_results = model.predict(([ my_story, my_ques]))
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])



