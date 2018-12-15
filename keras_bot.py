from classcized_dataset import BaseModel
from random import randint
from keras import layers
from keras.models import model_from_yaml, Sequential, load_model
from keras.optimizers import RMSprop
import keras 
import numpy as np
import os
import json
import sys

class KerasModel(BaseModel):


    def __init__(self):
        self.model_descriptor_yaml = None
        self.model_name = None#"init_char_RNN.h5" #ensure input maxlen compatibiltiy before import
        self.user_feed = True
        self.chars = None
        self.words = None
        self.maxlen = 10
        self.step = 5
        self.model_input_type = "char" # Also accepts word for now#needs to be attached to model_desc + model_name
        self.xs = []
        self.ys = []
        self.mc = keras.callbacks.ModelCheckpoint('newweights{epoch:08d}.h5', 
                                     save_weights_only=True, period=50)
        self.inputize()
        self.vectorize()
        self.model = self.generate_model()
        
    def generate_model(self):
        if self.model_descriptor_yaml:
            # If model descriptor is existant
            self.model = model_from_yaml(self.model_descriptor_yaml)
            return self.model
        #elif self.model_name:
        #    self.model = load_model(self.model_name)
        elif self.model_input_type == "char":   
            # If model yaml not found, build default lstm model
            self.model = Sequential()
            self.model.add(layers.LSTM(128, input_shape=(self.maxlen, len(self.chars))))
            self.model.add(layers.Dense(len(self.chars), activation='softmax'))
            optimizer = RMSprop(lr=0.01)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            self.model.load_weights("weights00000050.h5")
            return self.model
        elif self.model_input_type == "word":
            # If model yaml not found, build default lstm model
            self.model = Sequential()
            self.model.add(layers.LSTM(128, input_shape=(self.maxlen, len(self.words))))
            self.model.add(layers.Dense(len(self.words), activation='softmax'))
            optimizer = RMSprop(lr=0.01)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            return self.model            

    def inputize(self):
        if self.model_input_type == "char":
            print("Generating char input lists")
            for i in range(0, len(self.aggregate_user_corpus)-self.maxlen, self.step):
                self.xs.append(self.aggregate_user_corpus[i:i+self.maxlen])
                self.ys.append(self.aggregate_user_corpus[i+self.maxlen])

            print( "Number of char sequences " +  str(len(self.xs)) )
            self.chars = sorted(list(set(self.aggregate_user_corpus)))
            self.char_dict = dict((char, self.chars.index(char)) for char in self.chars)

        elif self.model_input_type == "word":
            self.word_user_corpus = self.aggregate_user_corpus.split(" ")
            for i in range(0, len(self.word_user_corpus)-self.maxlen, 1):
                self.xs.append(self.aggregate_user_corpus[i:i+self.maxlen])
                self.ys.append(self.aggregate_user_corpus[i+self.maxlen])

            print( "Number of word sequences " +  str(len(self.xs)) )
            self.words = sorted(list(set(self.word_user_corpus)))
            print( "Number of words " +  str(len(self.words)) )
            self.word_dict = dict((word, self.words.index(word)) for word in self.words)

    def vectorize(self):
        if self.model_input_type == "char":
            print("Generating input vectors")
            self.x_vec = np.zeros( (len(self.xs),self.maxlen,len(self.chars)), dtype=np.bool)
            self.y_vec = np.zeros( (len(self.xs),len(self.chars)), dtype=np.bool )
            for i, sentence in enumerate(self.xs):
                for t, char in enumerate(sentence):
                    self.x_vec[i, t, self.char_dict[char]] = 1
                    self.y_vec[i, self.char_dict[self.ys[i]]] = 1
        
        elif self.model_input_type == "word":
            self.x_vec = np.zeros( (len(self.xs),self.maxlen,len(self.words)), dtype=np.bool)
            self.y_vec = np.zeros( (len(self.xs),len(self.words)), dtype=np.bool )

            for i, sentence in enumerate(self.xs):
                for t, word in enumerate(sentence):
                    self.x_vec[i, t, self.word_dict[word]] = 1
                    self.y_vec[i, self.word_dict[self.ys[i]]] = 1
    
    @staticmethod
    def _sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
            
    def predict_sequence(self):
        if self.user_feed:
            start_statement = input("Send to bot:")
            generated_text = "" + start_statement

            for temprature in [1.0]:
                print(" @ Temp: "+ str(temprature) )
                
                sys.stdout.write(generated_text)
                for i in range(200): #Add a condition while output != \n
                    sampled = np.zeros((1, self.maxlen, len(self.chars)))
                    for t, char in enumerate(generated_text):
                        sampled[0, t, self.char_dict[char]] = 1
                    
                    preds = self.model.predict(sampled, verbose=0)[0]
                    next_index = self._sample(preds, temprature)
                    next_char = self.chars[next_index]
                    generated_text += next_char
                    generated_text = generated_text[1:]
                    sys.stdout.write(next_char)

    def train_loop(self):
        raise NotImplementedError