# Decode MEG data at every time point and get rank of prediction
# Pipeline:
# FOR every sub time window:
		# FOR all possible leave-one-out permutations:
			# leave one out, train on the rest
			# test on the left-out word: get predicted vector
			# assess score using rank = cos similarity to all the words and choose the rank of the actual target.
			# save the rank in a list.

		# repeat using all leave-one out combinations.
		# final score is mean rank.

# repeat at all time windows: one (mean) rank score per time window


import pickle,os, mne
import numpy as np
import pandas as pd
import scipy.stats as ss
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
# evoked = averages of repeats, within the Naming and Reading tasks.

log['target'] = [i.replace('.jpg', '') for i in log.target]
# words = np.unique(log.target)
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
# X_naming = evoked_images
# X_reading = evoked_words
X = evoked_words
y = embeddings_aligned
print('X_naming shape:%s\nX_reading shape:%s\ny shape: %s' %(X_naming.shape, X_reading.shape, y.shape))


# ========== TEMPORAL DECODING
loo = LeaveOneOut()
all_ranks = [] # shape is [n_splits, n_times] one rank pred per crossval split per time

for train_index, test_index in loo.split(X): # leave one out splits. All possible combinations

    # Train-test split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create classifier
    clf = make_pipeline(StandardScaler(), RidgeCV())
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='neg_mean_squared_error', verbose=True)

    # fit to training data, one fit per time point
    time_decod.fit(X_train,y_train)
    # predict test data: one prediction per time-point
    y_test_predict = time_decod.predict(X_test)

    ranks = []
    for timepoint in range(len(epochs.times)): # for every time point
        cosine_sims = [] # to append all the cos similiartities at the current time point.
        y_pred_at_timepoint = y_test_predict[:,timepoint,:] # grab the precition at current time point

        # next get cosine similarity of precition with all the vectors
        for vec in y: # for every vector
            vec = vec.reshape(1,-1) # add a dimension, needed for cosine_similarity
            cos = 1 - cosine_similarity(vec ,y_pred_at_timepoint)[0][0]

            cosine_sims.append(cos)

        # get the rank of the real vector from the cosine sims witht the predicted y
        rank = ss.rankdata(cosine_sims)[test_index][0]

        ranks.append(rank)

    all_ranks.append(ranks)

all_ranks = np.array(all_ranks)
mean_ranks = all_ranks.mean(axis=0)
stdev_ranks = all_ranks.std(axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, mean_ranks, label='mean rank', color='blue')
ax.fill_between(epochs.times, mean_ranks-stdev_ranks, mean_ranks+stdev_ranks)

ax.set_xlabel('Times')
ax.set_ylabel('Model prediction rank')
ax.legend()
ax.set_title('Sensor space decoding of BERT embeddings')
