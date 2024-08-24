## This directory generates behavior figures that don't require video
# main5 plots result of whisker trim

import datetime
import numpy as np
import pandas
import scipy.stats
import matplotlib.pyplot as plt
import my.plot
import os
import pytz


## Setup
# Figures
my.plot.manuscript_defaults()
my.plot.font_embed()

# Paths
load_path = os.path.abspath('../path_to_downloaded_data')
try:
    with open(load_path) as fi:
        repository_dir = fi.readlines()[0].strip()
except FileNotFoundError:
    raise IOError(
        'you need to download the data and specify its location in ' + 
        load_path)

if not os.path.exists(repository_dir):
    raise IOError(
        'downloaded data must exist in directory {}'.format(repository_dir))
    
    
## Load data from main1
mouse_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'mouse_data'))
perf_metrics = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'perf_metrics'))
session_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'session_data'))
trial_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'trial_data'))
poke_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'poke_data'))


## Localize the whisker trim date
tz = pytz.timezone('America/New_York')

# TODO: move this to 00_preprocessing/main1d
dt_like_col = 'whisker trim date'

# Convert from string to pandas.timestamp (this is probably unnecessary)
mouse_data[dt_like_col] = pandas.to_datetime(mouse_data[dt_like_col])

# Localize
mouse_data[dt_like_col] = mouse_data[dt_like_col].dt.tz_localize(tz)

# Choose a time
mouse_data[dt_like_col] += datetime.timedelta(hours=17)


## Align performance to whisker trim
# This is where we will keep track of n_sessions from trim date
session_data['n_from_trim'] = np.nan

# Align to manipulation date
for mouse, subdf in session_data.groupby('mouse'):
    # Skip if no manipulation
    if pandas.isnull(mouse_data.loc[mouse, 'whisker trim date']):
        continue
    
    # Slice out sessions after manipulation
    post_manipulation_sessions = subdf[
        subdf['first_trial'] > mouse_data.loc[mouse, 'whisker trim date']
        ]
    
    # Warn if no post-manipulation sessions
    if len(post_manipulation_sessions) == 0:
        print('warning: no post-manipulation sessions for {}'.format(mouse))
    
    else:
        # Get the n_session of the first post-manipulation session
        manipulation_n_session = post_manipulation_sessions['n_session'].min()
        
        # Align to this
        # So 0 will be the first session after the manipulation
        n_from_manipulation = (
            subdf['n_session'] - manipulation_n_session)
        
        # Store
        session_data.loc[
            n_from_manipulation.index, 'n_from_trim'
            ] = n_from_manipulation.values

# Join n_from_trim onto perf_metrics
perf_metrics = perf_metrics.join(
    session_data['n_from_trim'], on=['mouse', 'session_name'])

# Drop sessions from mice that never got trimmed
perf_metrics = perf_metrics.dropna(subset=['n_from_trim'])

# Drop mice that never got trimmed
mouse_data = mouse_data.dropna(subset=['whisker trim date'])


## Select rcp from around the trim
# Select rcp indexed by n_from_trim
perf_by_mouse = perf_metrics.reset_index().set_index(
    ['mouse', 'n_from_trim'])['rcp'].unstack('mouse')

# Slice around trim
perf_by_mouse = perf_by_mouse.loc[-3:0]
assert not perf_metrics.isnull().any().any()


## Aggregate
pre3 = perf_by_mouse.loc[-3:-1].mean()
pre1 = perf_by_mouse.loc[-1]
post = perf_by_mouse.loc[0]
agg = pandas.concat(
    [pre3, pre1, post], keys=['pre3', 'pre1', 'post'], 
    axis=1, verify_integrity=True)

# Convert to ppt
agg += 1


## Join on hl type
agg = agg.join(
    mouse_data['hearing loss type']).set_index(
    'hearing loss type', append=True).swaplevel().sort_index()


## Stats
pre3_pval = scipy.stats.ttest_rel(agg['pre3'], agg['post']).pvalue
pre1_pval = scipy.stats.ttest_rel(agg['pre1'], agg['post']).pvalue


## Plot
for pre_col in ['pre1']: #'pre3', 'pre1']:
    f, ax = plt.subplots(figsize=(2.5, 3))
    f.subplots_adjust(left=.2, bottom=.2, top=.93, right=.95)
    ax.plot(
        agg.loc['bilateral', [pre_col, 'post']].T, 
        marker='o', ls='-', mfc='none', color='r')
    ax.plot(
        agg.loc['left', [pre_col, 'post']].T, 
        marker='o', ls='-', mfc='none', color='b')
    ax.plot(
        agg.loc['right', [pre_col, 'post']].T, 
        marker='o', ls='-', mfc='none', color='b')
    ax.plot(
        agg.loc['sham', [pre_col, 'post']].T, 
        marker='o', ls='-', mfc='none', color='k')
    ax.set_ylim((1, 4.5))
    ax.set_yticks((1, 2, 3, 4))
    ax.set_xlim((-0.5, 1.5))
    my.plot.despine(ax)
    
    ax.set_xticks((0, 1))
    ax.set_xticklabels(('pre', 'post'))
    ax.set_xlabel('whisker trim')
    ax.set_ylabel('ports poked per trial\n(lower is better)')

    ax.plot([0, 1], [4.5, 4.5], lw=1, color='k', clip_on=False)
    ax.text(.5, 4.55, 'ns', va='bottom', ha='center')

plt.show()


## Save
f.savefig('WHISKER_TRIM.svg')
f.savefig('WHISKER_TRIM.png', dpi=300)


## Save stats
with open('STATS__WHISKER_TRIM', 'w') as fi:
    fi.write('n = {} mice\n\n'.format(len(agg)))
    fi.write('n by type:\n{}\n\n'.format(
        agg.groupby('hearing loss type').size()))
    fi.write('n by cohort:\n{}\n\n'.format(
        mouse_data.groupby(['cohort', 'sex']).size()))
    fi.write('pvalue on pre-day 1 vs post-day: {:.2g}\n\n'.format(pre1_pval))
    fi.write('days after surgery: {}\n\n'.format(
        mouse_data['whisker trim date'] - mouse_data['hearing loss date']))

with open('STATS__WHISKER_TRIM') as fi:
    print(''.join(fi.readlines()))
