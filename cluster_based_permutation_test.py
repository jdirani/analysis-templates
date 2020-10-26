# Spatio-temporal permutaiton test on MEG data (in source space).

# FOLDERS STRUCTURE:
# > OUT
#   > Results
#       > ObjectNaming
#       > WordReading


#====================Import data into eelbrain.Dataset=========================#
ROOT = '/Users/julien/Desktop/semint_workspace/Brain/Data/'
OUT = '/Users/julien/Desktop/semint_workspace/Brain/'

import mne, os, eelbrain, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

subjects_dir= ROOT + 'MRI/'
subjects =[
'A0031', 'A0129', 'A0134', 'A0155', 'A0182', 'A0205', 'A0229', 'A0283',
'A0285', 'A0287', 'A0288', 'A0289', 'A0290', 'A0291', 'A0292', 'A0293',
'A0294','A0295', 'A0297', 'A0300', 'A0301', 'A0302', 'A0303', 'A0304',
'A0209'
]

#Create/load fsaverage source space:
if not os.path.isfile(ROOT+'MRI/fsaverage/bem/fsaverage-ico-4-src.fif'):
    print "Createing fsaverage source space"
    src = mne.setup_source_space(subject='fsaverage', spacing='ico4', subjects_dir=subjects_dir)
    src.save(ROOT+'MRI/fsaverage/bem/fsaverage-ico-4-src.fif')
    del src


stcs, prime_type, task, soa, subject = [],[],[],[],[]
#---------------------------load stcs of all subjs-----------------------------#
for subj in subjects:
    stc_path = ROOT+'STC/%s/'%subj
    labels = [i for i in os.listdir(stc_path) if i.endswith('-lh.stc')]
    for i in labels:
        stc = mne.read_source_estimate(stc_path + i)
        stcs.append(stc)
        prime_type.append(str.split(i,'_')[1])
        task.append(str.split(i,'_')[2])
        soa.append(str.split(i,'_')[3])
        subject.append(subj)
        del stc

#--------------------Loading into an eelbrain.Dataset--------------------------#
ds = eelbrain.Dataset()
ds['stc'] = eelbrain.load.fiff.stc_ndvar(stcs,subject='fsaverage',src='ico-4',subjects_dir=subjects_dir,method='dSPM',fixed=False,parc='aparc')
ds['Task'] = eelbrain.Factor(task)
ds['Prime_Type'] = eelbrain.Factor(prime_type)
ds['SOA'] = eelbrain.Factor(soa)
ds['subject']=eelbrain.Factor(subject,random=True)

src=ds['stc'] #for convenience. Also to keep the full stcs saved in case I reassigned ds['stc'] to a subset (for a subregion)


#==============================================================================#
#                           Done importing data                                #
#==============================================================================#



#-----------------------cluster based permutation test-------------------------#
# condition names
SOAs = ['150','200','250','300']
Tasks = ['ObjectNaming', 'WordReading']
Prime_Types = ['id', 'semrel', 'unrel']
pvalue = 0.05

for current_task in Tasks:
    # Using full left hemisphere:
    ds['stc']=src #reset data to full space
    src.source.set_parc('FullLeftHemisphere') #choose atlas
    src_region = src.sub(source='left_hemi-lh') #reducing the ds to just the sources of interest. can also sub with time.
    ds['stc']=src_region

    res = eelbrain.testnd.anova('stc', X='Prime_Type*SOA',ds=ds,sub=ds['Task']==current_task ,match='subject',pmin=0.01, tstart=0.1, tstop=0.6, samples=10000, mintime=0.01, minsource=10)
    print res.clusters
    pickle.dump(res, open(OUT+'Results/%s/res.p'%current_task,'wb'))
    f=open(OUT+'Results/%s/results_table.txt'%current_task, 'w')
    f.write('Model: %s, N=%s\n' %(res.X, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()

    ix_sign_clusters=np.where(res.clusters['p']<=pvalue)[0]

    for i in range(len(ix_sign_clusters)):
        cluster = res.clusters[ix_sign_clusters[i]]['cluster']
        tstart = res.clusters[ix_sign_clusters[i]]['tstart']
        tstop = res.clusters[ix_sign_clusters[i]]['tstop']
        effect = res.clusters[ix_sign_clusters[i]]['effect']

        # Need to rename, else error
        if effect == 'Task x Prime_Type':
            effect = 'Task%Prime_Type'
            set_cluster = 'Task x Prime_Type'
        elif effect == 'Task x SOA':
            effect = 'Task%SOA'
            set_cluster = 'Task x SOA'
        elif effect == 'Prime_Type x SOA':
            effect = 'Prime_Type%SOA'
            set_cluster = 'Prime_Type x SOA'
        elif effect == 'Task x Prime_Type x SOA':
            effect = 'Task%Prime_Type%SOA'
            set_cluster = 'Task x Prime_Type x SOA'
        else:
            effect = effect
            set_cluster = effect

        #save significant cluster as a label for plotting.
        label = eelbrain.labels_from_clusters(cluster)
        label[0].name = 'label-lh'
        mne.write_labels_to_annot(label,subject='fsaverage', parc='cluster%s_FullAnalysis'%i ,subjects_dir=subjects_dir, overwrite=True)
        src.source.set_parc('cluster%s_FullAnalysis' %i)
        src_region = src.sub(source='label-lh')
        ds['stc']=src_region
        timecourse = src_region.mean('source')

        #plot
            # 1)Timecourse
        activation = eelbrain.plot.UTSStat(timecourse, effect, ds=ds, sub=ds['Task']==current_task, legend='lower left', title='cluster%s time course, Task=%s,effect=%s' %(i+1, current_task, effect))
        activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50)
        activation.save(OUT+'Results/%s/cluster%s_timecourse_(%s-%s)_effect=%s.png' %(current_task, i+1,tstart, tstop, effect))
        activation.close()
            # 2) Brain
        brain = eelbrain.plot.brain.cluster(cluster.mean('time'), subjects_dir=subjects_dir, surf='smoothwm')
        brain.save_image(OUT+'Results/%s/cluster%s_brain_(%s-%s)_Task=%s,effect=%s.png' %(current_task, i+1, tstart, tstop, current_task, effect))
        brain.close()
            # 3) Bar graph
        ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
        bar = eelbrain.plot.Barplot(ds['average_source_activation'], X=effect, ds=ds, sub=ds['Task']==current_task)
        bar.save(OUT+'Results/%s/cluster%s_BarGraph_(%s-%s)_Task=%s,effect=%s.png'%(current_task,i+1, tstart, tstop, current_task, effect))
        bar.close()



#==============================================================================#
#==============================================================================#
