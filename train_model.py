# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:38:54 2024

@author: ASUS
"""

#Run in Jupyter step by step
import music21 as m21
import os
import json
import tensorflow.keras as keras
import numpy as np
########################################
kern_dataset_path = 'C://Users//ASUS//Documents//Python Scripts//Trainning//experiments//music generation//EDM//rock'#where u have the midi files for trainning
m21.environment.set("musescoreDirectPNGPath", "C:/Program Files/MuseScore 4/bin/MuseScore4.exe")
acceptable_duration = [0.25, 0.5, 0.75, 1, 1.5, 2,3,4]
SAVE_DIR = "datasetrock"
single_file_dataset = "file_datasetrock"
SEQUENCE_LENGTH = 64
mapping_path = "mappingrock.json"  #how and where u save the notes
#######################################
def load_songs_in_kern(dataset_path):
    songs = []   

    for path,subdirs,files in os.walk(dataset_path):
        for file in files:
            if file[-3:]=="mid":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
                
    return songs
############################################
def has_acceptable_duration(song, acceptable_duration):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True
############################################
def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    #print(key)
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode =="minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
        
    transposed_song = song.transpose(interval)
    return transposed_song
####################################
def encode_song(song, time_step = 0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song
######################################
def preprocess(dataset_path):
    pass
    print("loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")
    
    for i,song in enumerate(songs):
        if not has_acceptable_duration(song, acceptable_duration):
            continue
        song = transpose(song)
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
            
###############################################
def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song
###########################################
def create_single_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs
###################################

#this willverify the amount of songs u parsed
if __name__ == "__main__":
    songs = load_songs_in_kern(kern_dataset_path)
    #print(f"loaded {len(songs)}  songs")
    song = songs[2]
  
    preprocess(kern_dataset_path)
    songs = create_single_dataset(SAVE_DIR,single_file_dataset,SEQUENCE_LENGTH)
   
    transposed_song = transpose(song)
    transposed_song.show()
#######################################
def create_mapping(songs, mapping_path):
    mappings = {}
    songs = songs.split()
    vocabulary = list(set(songs))
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent = 4)
############################################
def convert_songs_int(songs):
    int_songs = []
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs
##########################################
def generate_training_sequences(sequence_length):
    songs = load(single_file_dataset)
    int_songs = convert_songs_int(songs)
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
    targets = np.array(targets)
    return inputs, targets
####################################
def main():
    preprocess(kern_dataset_path)
    songs = create_single_dataset(SAVE_DIR,single_file_dataset,SEQUENCE_LENGTH)
    create_mapping(songs, mapping_path)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
if __name__ == "__main__":
    main()
#####################################
import tensorflow.keras as keras

OUTPUT_UNITS = 43 #check generated json file for the number of notes
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 70
BATCH_SIZE = 32
SAVE_MODEL_PATH = "modelrock.h5"    #name the model
#####################################################
def build_model(output_units, num_units, loss, learning_rate):
    input = keras.layers.Input(shape = (None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    
    output = keras.layers.Dense(output_units, activation = "softmax")(x)
    
    model = keras.Model(input, output)
    
    model.compile(loss = loss, optimizer = keras.optimizers.Adam(lr = learning_rate), metrics = ["accuracy"])
    model.summary()
    return model
#########################################################
def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS , loss = LOSS, learning_rate = LEARNING_RATE):
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    model = build_model(output_units, num_units, loss, learning_rate)
    model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)
    model.save(SAVE_MODEL_PATH)
######################################################
if __name__ == "__main__":
    train()
    
#######################################