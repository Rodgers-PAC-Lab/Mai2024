## Plot quantifications of startle

import pandas
import numpy as np
import my.plot
import scipy.stats
import matplotlib.pyplot as plt
import os

## Plotting params
my.plot.manuscript_defaults()
my.plot.font_embed()


## Paths
# Data repository
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

# Put output data here
startle_data_dir = os.path.join(repository_dir, 'startle')


## Load metadata about mice that includes hearing loss date
mouse_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'mouse_data'))


## Load data from main1*
startle_metadata = pandas.read_pickle(
    os.path.join(startle_data_dir, 'startle_metadata'))
speed_locked_to_onsets = pandas.read_pickle(
    os.path.join(startle_data_dir, 'speed_locked_to_onsets'))

# Get the mouse names
mouse_names = sorted(startle_metadata['mouse'].unique())

# Why is this called video instead of video_filename?
startle_metadata.index.name = 'video_filename'

# Standardize the mouse names
startle_metadata['mouse'] = startle_metadata['mouse'].replace({
    'Butterfly090': 'Butterfly_090',
    'Butterfly091': 'Butterfly_091',
    'Guitar089': 'Guitar_089',
    'KiddingOctopus085': 'Kidding_Octopus_085',
    'KiddingOctopus086': 'Kidding_Octopus_086',
    'KiddingOctopus087': 'Kidding_Octopus_087',
    'PartyOwl095': 'Party_Owl_095',
    'PartyOwl096': 'Party_Owl_096',
    'PartyOwl097': 'Party_Owl_097',
    'Watermelon092': 'Watermelon_092',
    'Watermelon093': 'Watermelon_093',
    'Watermelon094': 'Watermelon_094',
    })


## Calculate days from hearing loss
# Join on hearing loss date
startle_metadata = startle_metadata.join(
    mouse_data[['hearing loss date', 'manipulation2']], on='mouse')

# We use business days here because of holidays and weekends
# These should map onto "behavioral sessions", if not real days
startle_metadata['dphl'] = startle_metadata.apply(
    lambda row: np.busday_count(row['hearing loss date'].date(), row['date']), 
    axis=1)


## Join the mouse name and hearing loss info from metadata to speed_locked_to_onsets
speed_locked_to_onsets = my.misc.join_level_onto_index(
    speed_locked_to_onsets, 
    startle_metadata[['mouse', 'dphl', 'manipulation2']])

# Drop video_filename (redundant with dphl)
speed_locked_to_onsets = speed_locked_to_onsets.droplevel('video_filename')


## Join startle volume onto speed_locked_to_onsets
# This is the startle volumes, copied from paft_startle.py
startle_volumes = pandas.Series([
    0.01,  0.003, 0.003, 0.01,  0.01,  0.003,
    0.01,  0.01,  0.003, 0.003, 0.003, 0.01,
    0.003, 0.01,  0.01,  0.01,  0.003, 0.003,
    0.01,  0.003, 0.003, 0.01,  0.003, 0.01,
    0.003, 0.003, 0.01,  0.003, 0.003, 0.01,
    0.01,  0.003, 0.01,  0.01,  0.01,  0.01,
    0.003, 0.01,  0.003, 0.003, 0.01,  0.003,
    ])
startle_volumes.index.name = 'onset'
startle_volumes.name = 'volume'

# Join volume on speed_locked_to_onsets
speed_locked_to_onsets = my.misc.join_level_onto_index(
    speed_locked_to_onsets, startle_volumes.to_frame(), put_joined_first=False)

# Reorder levels
speed_locked_to_onsets = speed_locked_to_onsets.reorder_levels(
    ['manipulation2', 'mouse', 'dphl', 'volume', 'onset', 'frame']
    ).sort_index()


## Finalize speed_locked_to_onsets
# Drop the first onsets, which is often an outlier
speed_locked_to_onsets = speed_locked_to_onsets.drop(0, level='onset')

# droplevel volume because we need the trials more than the levels
# The volume 0.10 startle is about 25% bigger but otherwise similar
# Using the soft volume only is slightly cleaner
# Using the loud volume only is too noisy
speed_locked_to_onsets = speed_locked_to_onsets.droplevel('volume').sort_index()

# Convert px/frame to px/s
speed_locked_to_onsets = speed_locked_to_onsets * 30

# Conver px/s to cm/s
speed_locked_to_onsets = speed_locked_to_onsets * 1.4 / 10


## Mean over body parts
startle_by_frame = speed_locked_to_onsets.mean(1).unstack('frame')



## Aggregate over trials first, then quantify
# median over onsets
startle_by_frame_meantrial = startle_by_frame.groupby(
    ['manipulation2', 'mouse', 'dphl']).median()

# aggregate over trials by operating directly on startle_by_frame_meantrial
agged_startle = pandas.DataFrame.from_dict({
    'post_mean2': startle_by_frame_meantrial.loc[:, 0:5].mean(axis=1),
    })


## Plots
PRETTY_PLOT_ALL_DAYS_MEAN_BY_HL_GROUP = True
PRETTY_PLOT_SUMMARY2 = True


## Same as PRETTY_PLOT_ALL_MICE_ALL_DAYS but with error bars instead of individual mice
if PRETTY_PLOT_ALL_DAYS_MEAN_BY_HL_GROUP:
    # only show the days with all mice: -1, 1, 5 (and maybe not even 1)
    plot_days = [-1, 1, 5]
    hl_types = ['hl_sham', 'hl_unilateral', 'hl_bilateral', ]

    f, axa = plt.subplots(len(hl_types), len(plot_days), sharex=True, sharey=True, figsize=(6.9, 6))
    f.subplots_adjust(left=.12, right=.87, top=.95, bottom=.13, hspace=.4, wspace=.4)
    f.text(.02, .5, 'movement speed (cm/s)', ha='center', va='center', rotation=90)
    f.text(.93, .82, 'sham\nn=4', ha='center', va='center')
    f.text(.93, .55, 'unilateral\nn=4', ha='center', va='center')
    f.text(.93, .25, 'bilateral\nn=4', ha='center', va='center')
    f.text(.48, .05, 'time after startle stimulus (s)', ha='center', va='center')

    for hl_type in hl_types:
        these_mice = startle_by_frame_meantrial.loc[hl_type]
        these_mice_names = these_mice.index.get_level_values('mouse').unique()
        for plot_day in plot_days:
            # Get this dphl
            topl_all_mice = these_mice.xs(plot_day, level='dphl')
            
            # Reindex by mouse to get the ordering consistent
            topl_all_mice = topl_all_mice.reindex(these_mice_names)
            
            # Plot
            ax = axa[
                hl_types.index(hl_type),
                plot_days.index(plot_day),
                ]

            ax.plot(topl_all_mice.columns.values / 30., topl_all_mice.mean(), color='k')
            ax.fill_between(
                x=topl_all_mice.columns.values / 30.,
                y1=topl_all_mice.mean() - topl_all_mice.sem(),
                y2=topl_all_mice.mean() + topl_all_mice.sem(),
                alpha=.5, color='k',
                lw=0)
            
            # Pretty
            if ax in axa[:, 0]:
                my.plot.despine(ax)
            else:
                my.plot.despine(ax, which=('left', 'top', 'right'))
            
            # title by plot_day
            if ax in axa[0]:
                if plot_day == -1:
                    ax.set_title('1d pre')
                elif plot_day == 1:
                    ax.set_title('1d post')
                elif plot_day == 5:
                    ax.set_title('5d post')
                else:
                    ax.set_title(str(plot_day))

    ax.set_ylim((0, 35))
    ax.set_yticks((0, 10, 20, 30))
    ax.set_xlim((-.5, 1))
    ax.set_xticks((0, 1))

    f.savefig('PRETTY_PLOT_ALL_DAYS_MEAN_BY_HL_GROUP.png', dpi=300)
    f.savefig('PRETTY_PLOT_ALL_DAYS_MEAN_BY_HL_GROUP.svg')  


if PRETTY_PLOT_SUMMARY2:
    ## Plot the summaries
    f, axa = plt.subplots(3, 1, figsize=(2.5, 6), sharex=True)
    f.subplots_adjust(left=.28, right=.95, top=.95, bottom=.13, hspace=.4, wspace=.4)
    colname = 'post_mean2'
        
    this_colname = agged_startle.loc[:, colname].unstack('dphl').loc[:, [-1, 1, 5]]
    hl_types = ['hl_sham', 'hl_unilateral', 'hl_bilateral', ]
    
    f.text(
        .05, .5, 'mean movement during startle (cm/s)', 
        ha='center', va='center', rotation=90)
    
    stats_res_l = []
    stats_res_keys_l = []
    for hl_type in hl_types:
        ax = axa[hl_types.index(hl_type)]
    
        topl = this_colname.loc[hl_type]
        ax.plot(topl.T.values, color='gray', lw=1, mfc='none', ms=4)
        my.plot.despine(ax)
    
        ax.plot(topl.mean().values, color='k', lw=3)#, marker='o', mfc='none')
    
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([-1, 1, 5])
        ax.set_xlim((-0.5, 2.5))
        ax.set_ylim((0, 35))
        ax.set_yticks((0, 10, 20, 30))
        
        # Run stats
        # Paired t-test between every day for this hl group
        rec_l = []
        for day1 in topl.columns:
            for day2 in topl.columns:
                if day1 >= day2:
                    continue
                ttest_res = scipy.stats.ttest_rel(topl.loc[:, day1], topl.loc[:, day2])
                pvalue = ttest_res.pvalue
                rec_l.append((day1, day2, pvalue))
        stats_res = pandas.DataFrame.from_records(
            rec_l, columns=['day1', 'day2', 'p']).set_index(['day1', 'day2'])['p']
        
        # Append
        stats_res_l.append(stats_res)
        stats_res_keys_l.append(hl_type)
        
        # Plot pval
        lower_line = 30
        upper_line = 34.5
        
        ax.plot([0.2, 0.8], [lower_line, lower_line], 'k-', clip_on=False, lw=1)
        ax.plot([1.2, 1.8], [lower_line, lower_line], 'k-', clip_on=False, lw=1)
        ax.plot([0.2, 1.8], [upper_line, upper_line], 'k-', clip_on=False, lw=1)
        
        # first comparison
        if stats_res.loc[-1].loc[1] < .05:
            ax.text(0.5, lower_line - 1.5, '*', ha='center', va='bottom')
        else:
            ax.text(0.5, lower_line, 'n.s.', ha='center', va='bottom', size=12)

        # second comparison
        if stats_res.loc[1].loc[5] < .05:
            ax.text(1.5, lower_line - 1.5, '*', ha='center', va='bottom')
        else:
            ax.text(1.5, lower_line, 'n.s.', ha='center', va='bottom', size=12)

        # third comparison (day -1 vs day 5)
        if stats_res.loc[-1].loc[5] < .05:
            ax.text(1, upper_line - 1.5, '*', ha='center', va='bottom')
        else:
            ax.text(1, upper_line, 'n.s.', ha='center', va='bottom', size=12)
    
    axa[-1].set_xlabel('days after surgery')
    
    
    ## Save fig
    f.savefig('PRETTY_PLOT_SUMMARY2.png', dpi=300)
    f.savefig('PRETTY_PLOT_SUMMARY2.svg')


    ## Save stats
    big_stats = pandas.concat(stats_res_l, keys=stats_res_keys_l, names=['hl_type'], axis=1)
    
    with open('STATS__PRETTY_PLOT_SUMMARY2', 'w') as fi:
        fi.write('paired t-test on each group, compared over each pair of days\n')
        fi.write(str(big_stats) + '\n\n')
        fi.write('sample size:\n')
        fi.write(str(this_colname.groupby('manipulation2').size()) + '\n\n')
    
    # write back
    with open('STATS__PRETTY_PLOT_SUMMARY2') as fi:
        print(''.join(fi.readlines()))

    
plt.show()
