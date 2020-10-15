import pickle,os, mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import cosine_similarity
from mne.decoding import SlidingEstimator, cross_val_multiscore,GeneralizingEstimator

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



# Train the classifier on all reading trials
# and test on all naming trials
clf = make_pipeline(StandardScaler(),  RidgeCV())
time_gen = GeneralizingEstimator(clf, scoring='neg_mean_squared_error', n_jobs=1,
                                 verbose=True)



# Fit classifiers on the reading epochs
time_gen.fit(X=X_reading,
             y=y)

# Score on the epochs for naming
scores = time_gen.score(X=X_naming,
                        y=y)

# Plot results
fig, ax = plt.subplots(1)
im = ax.matshow(scores, cmap='RdBu_r', origin='lower')
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time and condition for BERT embeddings\n(Neg mean sq error)')
plt.colorbar(im, ax=ax)
plt.show()
