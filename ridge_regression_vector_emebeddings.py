# decode word embeddings/Word2vec AND/OR word frequency for the Naming and Reading tasks, seperately.
# Decodnig at every time point and plot accuracy using negative MSE


import pickle,os, mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import cosine_similarity
from mne.decoding import SlidingEstimator, cross_val_multiscore

#====== load the data
data_path = '/Users/julien/Desktop/sample_datasets/semint_sample/'

epochs = mne.read_epochs(data_path + 'epochs/A0182/A0182_200-epo.fif')

log = pd.read_csv(data_path + 'logs/A0182_log_mask.csv')
log = log[log['mask']==1][log['SOA']==200] # align epochs and logs

embeddings = pickle.load(open('/Users/julien/Desktop/sample_datasets/semint_sample/semint_cls_embeddings.p', 'rb'))
ling_char = pd.read_csv('/Users/julien/Desktop/sample_datasets/semint_sample/ling_char.csv')

# # remove trial if embedding is nan
# embeddings_mask = log.embeddings != np.nan
# log = log[embeddings_mask]
# epochs = epochs[embeddings_mask]

# just take 2 conditions
assert len(epochs) == log.shape[0]


# ===== FORMAT THE VARS
# evoked = averages of repeats, within naming and reading

log['target'] = [i.replace('.jpg', '') for i in log.target]
words = np.unique(log.target)
evoked_words = []
evoked_images = []

embeddings_aligned = [] # align to evoked created above
word_freq_aligned = []
rt_aligned = []
for idx, w in enumerate(words):
    print(w, idx+1, '/', len(words))
    mask_wordasword = np.array((log.target == w) & (log.target_type == 'word'))
    mask_wordasimage = np.array((log.target == w) & (log.target_type == 'image'))

    evoked_words.append(epochs[mask_wordasword].average().data) # average over repets of the same word (that was presented as a word)
    evoked_images.append(epochs[mask_wordasimage].average().data) # average over repets of the same word (that was presented as an image)

    embeddings_aligned.append(embeddings[w])
    word_freq_aligned.append(ling_char[ling_char['Word'] == w]['Log_Freq_HAL'].iloc[0]) # fetch the word freq


# convet to numpy array
embeddings_aligned = np.array(embeddings_aligned)
evoked_words = np.array(evoked_words)
evoked_images = np.array(evoked_images)
word_freq_aligned = np.array(word_freq_aligned)

# ===== Edit here if needed:
X_naming = evoked_images
X_reading = evoked_words
y = embeddings_aligned
print('X_naming shape:%s\nX_reading shape:%s\ny shape: %s' %(X_naming.shape, X_reading.shape, y.shape))


# ========== TEMPORAL DECODING
clf = make_pipeline(StandardScaler(), RidgeCV())
time_decod = SlidingEstimator(clf, n_jobs=1, scoring='neg_mean_squared_error', verbose=True)

scores_naming = cross_val_multiscore(time_decod, X_naming, y, cv=5, n_jobs=1) #len(X) for leave one out CV
scores_reading = cross_val_multiscore(time_decod, X_reading, y, cv=5, n_jobs=1)

# Mean scores across cross-validation splits
scores_naming = np.mean(scores_naming, axis=0)
scores_reading = np.mean(scores_reading, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores_reading, label='score: reading task', color='blue')
ax.plot(epochs.times, scores_naming, label='score: naming task', color='red')
# ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('neg_mean_squared_error')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding of BERT embeddings')
