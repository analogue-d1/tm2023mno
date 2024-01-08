'''
The script below loads dictionary of feature matrices calculated in FeatureExtraction.py script.
It also contains a function to concatenate these dictionaries to obtain data in format:
{'language': ndarray(N,512)} for mean and max pooling
{'language': ndarray(2*N,512)} for mean&std pooling
{'language': ndarray(q*N,512)} for quantile pooling
where N depends on number of extracted .wav files and q is number of quantile folds.
'''

import json
import numpy as np

def load_dict_from_json(file_path):

    def convert_np_arrays(obj):
        if isinstance(obj, list):
            return [convert_np_arrays(item) for item in obj]
        elif isinstance(obj, dict) and '__numpy_array__' in obj:
            # Convert the dictionary to a NumPy array
            return np.array(obj['__numpy_array__'])
        else:
            return obj

    with open(file_path, 'r') as json_file:
        loaded_dict = json.load(json_file, object_hook=convert_np_arrays)

    return loaded_dict

def concatenateFeatures(feature_dict):
    result = {}
    keys = list(feature_dict.keys())

    for i in range(0,len(keys)):
        to_concatenate = []
        feats = feature_dict[keys[i]]

        for j in range(0,len(feats)):

            if feats[j].ndim != 1:
                to_concatenate.append(feats[j]) # in case of quantiles and mean&std
                conc_vec = np.concatenate(to_concatenate, axis = 0)
                result[keys[i]] = conc_vec
            else:
                to_concatenate.append(feats[j].flatten()) #flatten if shape is (1,512) and not (512,)
                conc_vec = np.transpose(to_concatenate).T
                result[keys[i]] = conc_vec
    
    return result


#example
root = r'C:\\'

#load each pooled features
loaded_dict_means = load_dict_from_json(root)
loaded_dict_means_std = load_dict_from_json(root)
loaded_dict_max = load_dict_from_json(root)
loaded_dict_quantiles = load_dict_from_json(root)

#Show dimentions of extracted arrays on english language example
print(loaded_dict_means['en'][0].shape)
print(loaded_dict_means_std['en'][0].shape)
print(loaded_dict_max['en'][0].shape)
print(loaded_dict_quantiles['en'][0].shape)

#concatenate each pooled features
conc1 = concatenateFeatures(loaded_dict_means)
conc2 = concatenateFeatures(loaded_dict_means_std)
conc3 = concatenateFeatures(loaded_dict_max)
conc4 = concatenateFeatures(loaded_dict_quantiles)

#Show dimentions of concatenated arrays on english language example
print(conc1['en'].shape)
print(conc2['en'].shape)
print(conc3['en'].shape)
print(conc4['en'].shape)

print("DONE!")
