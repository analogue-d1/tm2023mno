''' 
The script below performs feature extraction of .wav files containing multiple languages and pooling
using statistical methods: mean, max, mean&std and quantiles and saves the results to .json file.  

'''

import glob
import os
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import librosa
import numpy as np
import json

'''
 Takes .wav files from your folders. Required path: languages_folder -> folders each assined to a given language 
 which name should correspond with key values, ex. when we want 'en' key for english language in dictionary, 
 the corresponding folder ought to be named 'en'. 
'''
def getWavFilesDict(root,languages):
  wav_dict = {}
  for i in range(0,len(languages)):
    os.chdir(root + languages[i]) 
    wav_files = glob.glob('*.wav')
    wav_dict[languages[i]] = wav_files 
  return wav_dict

'''
Takes dictionary in format of {language1 : [1.wav, 2.wav, ... ] , language2 : [1.wav, ... }, ...}
and changes waves to floating point signal vectors from your folders.
'''
def createLabeledSet(wav_data_dict, folder_path,  sr = 16000):
  keys = list(wav_data_dict.keys())

  wav_dict = {}
  wav_length_dict = {}

  for i in range(0,len(keys)):
    wavs_float_list = []
    wav_length_list = []
    wavs_str_list = wav_data_dict[keys[i]]

    for j in range(0,len(wavs_str_list)):
      wav_vector, _ = librosa.load(folder_path
                                   + keys[i] + wavs_str_list[j], sr = sr)
      wav_vector_length = len(wav_vector)
      wavs_float_list.append(wav_vector)
      wav_length_list.append(wav_vector_length)
      wav_dict[keys[i]] = wavs_float_list
      wav_length_dict[keys[i]] = wav_length_list
  return wav_dict, wav_length_dict

'''
Useful for obtaining feature vector zero-padding.
'''
def obtainMaxLength(wav_length_dict):
  max_length_list = []
  keys = list(wav_length_dict.keys())

  for i in range(0,len(keys)):
    lengths = wav_length_dict[keys[i]]
    max_length_list.append(max(lengths))

  max_length = max(max_length_list)
  return max_length

def loadExtractor(model_name):
  model = Wav2Vec2Model.from_pretrained(model_name) #baza pretreningowa nie ma wpływu na rozmiary wektora cech
  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
  return model, feature_extractor

'''
Takes dictionary in format of {language1: [ndarray1, ndarray2, ...] , language2 : ndarray1, ...}
and changes floating point signal vectors to arrays containing wav2vec2 features. ndarray shape: (N,512), 
N depends on length.
'''
def extractFeatures(wav_dict, model, feature_extractor, max_length, sr=16000):
  keys = list(wav_dict.keys())
  print(f'languages to extract: {keys}')
  feature_dict = {}

  for i in range(0,len(keys)):
    feature_list = []
    signals = wav_dict[keys[i]]
    print(f"Extraction begin: {keys[i]}")

    for j in range(0,len(signals)):
      norm_signal = feature_extractor(signals[j], return_tensors= "pt", sampling_rate = sr, do_normalize=True,  padding='max_length', max_length = max_length)

      with torch.no_grad():
        feature = model(norm_signal.input_values, output_hidden_states = False, output_attentions = False)
        feature = feature.extract_features
        feature = feature.numpy()
        feature = feature[0]
        feature_list.append(feature)
        feature_dict[keys[i]] = feature_list

  return feature_dict
'''
Pools mean and mean&std from a feature dict. Result are in format of {language1 : [ ndarray(512,), ... ], ...}
and {language1 : [ ndarray(2,512), ... ],}, were this 2 comes from mean AND std.
'''
def poolMeansAndSTD(feature_dict):
  means_dict = {}
  means_std_dict = {}
  keys = list(feature_dict.keys())

  for i in range(0,len(keys)):
    means_list = []
    mean_std_list = []
    features = features_dict[keys[i]]

    for j in range(0,len(features)):
      mean = np.mean(features[j], axis = 0)
      means_list.append(mean)
      means_dict[keys[i]] = means_list

      std = np.std(features[j], axis = 0)
      mean_std = np.transpose([mean, std]).T
      mean_std_list.append(mean_std)
      means_std_dict[keys[i]] = mean_std_list

  return means_dict, means_std_dict
'''
Pools max value from a feature dict. Results are in format of {language1 : [ ndarray(512,), ... ], ...}
'''
def poolMax(feature_dict):
  max_dict = {}
  keys = list(feature_dict.keys())

  for i in range(0,len(keys)):
    max_list = []
    features = features_dict[keys[i]]

    for j in range(0,len(features)):
      max = np.max(features[j], axis = 0)
      max_list.append(max)
      max_dict[keys[i]] = max_list

  return max_dict
'''
Pools quantiles froam a feature dict. Results are in format of {language1 : [ ndarray(q,512), ... ], ...} 
where q is number of folds of quantiles in range from 0 to  
'''
def poolQuantiles(feature_dict,q):
  quantile_dict = {}
  keys = list(feature_dict.keys())

  for i in range(0,len(keys)):
    quantile_list = []
    features = features_dict[keys[i]]

    for j in range(0,len(features)):
      quantiles = []
      
      for k in range(0,q):
        quantile = np.quantile(features[j], k/(q-1), axis = 0)
        quantiles.append(quantile)

      quan = np.transpose(quantiles).T
      quantile_list.append(quan)
      quantile_dict[keys[i]] = quantile_list
  
  return quantile_dict
   
def save_dict_to_json(file_path, dictionary):

    def convert_np_arrays(obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy_array__': obj.tolist()}
        elif isinstance(obj, list):
            return [convert_np_arrays(item) for item in obj]
        else:
            return obj

    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, default= convert_np_arrays)

#example

#extracted languages
labels = ['de','en','es','fi','fr','it','ja','la','tr','uk']

wav_files_dict = getWavFilesDict(r'C:\\Users\\mikol\\Desktop\\folder\\data\\', labels)
wav_dict, wav_length_dict = createLabeledSet(wav_files_dict)
model, feature_extractor = loadExtractor("facebook/wav2vec2-base-960h")
xam = obtainMaxLength(wav_length_dict)

features_dict = extractFeatures(wav_dict,model,feature_extractor, xam)
features_mean, features_means_stds = poolMeansAndSTD(features_dict)
features_max = poolMax(features_dict)
features_quantiles = poolQuantiles(features_dict, 5)

#Show dimentions of extracted arrays on english language example
print(features_mean['en'][0].shape)
print(features_means_stds['en'][0].shape)
print(features_max['en'][0].shape)
print(features_quantiles['en'][0].shape)

#set path and variable name
root = r'C:\\'

#Save arrays to .json
save_dict_to_json(root, features_mean)
save_dict_to_json(root, features_means_stds)
save_dict_to_json(root, features_max)
save_dict_to_json(root, features_quantiles)

print("DONE!")