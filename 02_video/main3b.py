## This directory uses synced video to analyze body movement before/after HL
# main3.py plots stats about trajectories over learning and peri-HL

import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import my
import my.plot
import extras


## Plotting
my.plot.font_embed()
my.plot.manuscript_defaults()


## Paths
# Repository dir
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


## Load behavior data
mouse_data = pandas.read_pickle(
    os.path.join(repository_dir, 'behavior', 'mouse_data'))
perf_metrics = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_video', 'perf_metrics'))
session_data = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_video', 'session_data'))
trial_data = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_video', 'trial_data'))
poke_data = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_video', 'poke_data'))


## Load results about crossing and kinematics
print("loading crossings and kinematics")
pokes_with_crossings = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_crossings', 'pokes_with_crossings'))
synced_big_crossings_df = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_crossings', 'synced_big_crossings_df'))
synced_big_mouse_location_df = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_crossings', 'synced_big_mouse_location_df'))
print("done loading")


## Add quantifications to kinematics and crossings
# The only "kinematics" we need are x, y, and region from location_df
# This variable name is now an anachronism
synced_big_kinematics_df = synced_big_mouse_location_df#[['x', 'y', 'region']]

# Add columns to crossings (direction of ports, duration, outcome, and n_pokes)
# n_pokes is important because it defines the type of crossing
synced_big_crossings_df = extras.add_columns_to_crossings(
    pokes_with_crossings, synced_big_crossings_df, trial_data)

# Add crossing number to synced_big_kinematics_df
# We do this so we can assess paths during crossings later
# And potentially also so we can blank out PRP time
synced_big_kinematics_df = extras.add_crossing_number_to_kinematics(
    synced_big_crossings_df, synced_big_kinematics_df)

# Quantify distance traveled and straightness for each center crossing
synced_big_crossings_df = extras.quantify_paths_during_crossings(
    synced_big_crossings_df, synced_big_kinematics_df, session_data)


## Compute kinematics_by_trial (distance, straightness, time spent in regions)
# Compute time in each region
quantified_time_by_trial = extras.compute_time_in_each_region(
    synced_big_kinematics_df)

# Aggregate straightness within each trial (mean over crossings)
straightness_in_center = synced_big_crossings_df.groupby(
    ['session_name', 'trial', 'region_after']
    )['straightness'].mean().unstack('region_after').loc[:, 8].rename(
    'straightness_in_center')

# Aggregate distance travelled in center (sum over crossings)
total_distance_in_center = synced_big_crossings_df.groupby(
    ['session_name', 'trial', 'region_after']
    )['dist_travelled'].sum().unstack('region_after').loc[:, 8].rename(
    'dist_in_center')

# Join quantified_time_by_trial, straightness, and distance
kinematics_by_trial = quantified_time_by_trial.join(
    straightness_in_center).join(total_distance_in_center)

# Calculate mean speed in center
kinematics_by_trial['speed_in_center'] = (
    kinematics_by_trial['dist_in_center'] / 
    kinematics_by_trial['time_in_center']
    )

# Note that the straightness and distance in center can be null even
# when there is some time spent there. This is because the paths are 
# attributed to the trial on which they started. So if the trial starts
# while the mouse is already in the center, that kinematic will be attributed
# to the previous trial
print('warning: no straightness on {:.4f} of trials'.format(
    kinematics_by_trial['straightness_in_center'].isnull().mean()))

# However, time_in_center should never be null
assert not kinematics_by_trial[
    ['time_in_center', 'time_in_sides', 'time_total']].isnull().any().any()


## Extract center-out crossings
# Extract center outs
center_outs = extras.extract_center_out_crossings(synced_big_crossings_df)

# Categorize center-out-crossings
crossing_counts_by_trial = extras.categorize_center_out_crossings(center_outs)

# Error check trials without center outs
# Join n_center_outs on trial_data
trial_data = trial_data.join(crossing_counts_by_trial['n_center_outs'])
trial_data['n_center_outs'] = trial_data['n_center_outs'].fillna(0).astype(int)

# Error check: how many trials lack center-outs?
# This could happen for several reasons
#   - Video started after trial (in which case start_frame is null)
#   - Video cut off during the trial (no easy way to check this here)
#   - They entered the correct chamber during the ITI. If so this should be
#     a correct trial. To fix this, relabel center-outs based on reward time.
#   - They hopped directly over a wall.
# As long as these events are pretty rare, we can ignore it.

# Count trials with video
trials_with_video = trial_data[~trial_data['start_frame'].isnull()]

# Count trials with video but not center outs
trials_with_video_but_not_center_outs = trials_with_video[
    trials_with_video['n_center_outs'] == 0]

# Warn
print("warning: no center-outs on {}/{} of trials with video".format(
    len(trials_with_video_but_not_center_outs),
    len(trials_with_video)
    ))


## Concat the crossing metrics with the trajectory metrics
# First fillna crossing counts with zeros on trials with no crossings
to_join = crossing_counts_by_trial.reindex(
    kinematics_by_trial.index).fillna(0).astype(int)

# Join
metrics_by_trial = pandas.concat(
    [to_join, kinematics_by_trial], axis=1).sort_index()

bad_mask = metrics_by_trial['dist_in_center'].isnull()
print('warning: {}/{}={:.4f} trials with null dist_in_center'.format(
    bad_mask.sum(), len(bad_mask), bad_mask.sum() / len(bad_mask)))


## Exclude trials with very high number of crossings
# Identify trials with <100 center outs
keep_mask = metrics_by_trial['n_center_outs'] < 100

# Apply mask to center_outs
metrics_by_trial = my.misc.slice_df_by_some_levels(
    metrics_by_trial, keep_mask.index[keep_mask.values])

# Warn
print("dropping {:0.5f} of trials with too many crossings".format(
    1 - keep_mask.mean()))


## Aggregate these per-trial crossing metrics within each session
# Note these are in units of "crossings of each type per trial"
# not "fraction of trials with at least one of these crossing types"
metrics_by_session = metrics_by_trial.groupby('session_name').mean()
assert not metrics_by_session.isnull().any().any()

# Aggregate p_cycling by summing over trials
metrics_by_session['p_cycling'] = (
    metrics_by_trial.groupby('session_name')['n_cycling'].sum() /
    metrics_by_trial.groupby('session_name')['n_center_outs'].sum()
    )

# Quantify as fraction of trials with a return
metrics_by_session['frac_returned'] = (
    metrics_by_trial['into_prp'] > 0).groupby('session_name').mean()


## Index one copy by n_session and the other by n_from_manipulation
# Add manipulation to session_data
to_join = session_data.join(mouse_data['manipulation2']).reset_index('mouse')

# Put on index
metrics_wrt_start = my.misc.join_level_onto_index(
    metrics_by_session, 
    to_join[['manipulation2', 'mouse', 'n_session']],
    ).droplevel('session_name')

# Put on index
metrics_wrt_hl = my.misc.join_level_onto_index(
    metrics_by_session, 
    to_join[['manipulation2', 'mouse', 'n_from_manipulation']],
    ).droplevel('session_name')
    
# Drop null sessions (those from mice without manipulation)
metrics_wrt_hl = metrics_wrt_hl.drop(np.nan, level='n_from_manipulation')


## Plots
PLOT_HISTOGRAM_OF_SPEED_BY_REGION = True
QUANTIFY_CENTER_TRANSITS_BY_HL_TYPE = True
QUANTIFY_ENTRIES_THROUGHOUT_LEARNING_AND_HL = True

# Plot colors
group2color = {
    'hl_bilateral': 'r',
    'hl_unilateral': 'b',
    'hl_left': 'green',
    'hl_right': 'magenta',
    'hl_sham': 'gray',
    }


if PLOT_HISTOGRAM_OF_SPEED_BY_REGION:
    ## Histogram the speed, rather than the trial duration
    
    ## Drop Coffee_078 and Coffee_079
    # We have learning data for them, but not post-HL
    # Drop for compatibility with the rest of the analyses
    this_synced_big_kinematics_df = synced_big_kinematics_df.copy()
    drop_sessions = session_data.loc[
        ['Coffee_078', 'Coffee_079']].index.get_level_values('session_name')
    
    # Drop
    this_synced_big_kinematics_df = this_synced_big_kinematics_df.drop(
        drop_sessions)
    
    
    ## Plot histogram of trial duration
    # Identify how many sessions and mice were included
    included_sessions = this_synced_big_kinematics_df.index.get_level_values(
        'session_name').unique()
    
    # Slice session_data accordingly
    # This will error if a session is not found
    sliced = session_data.loc[pandas.IndexSlice[:, included_sessions], :]
    
    # Actually all sessions are included
    assert sliced.equals(session_data.drop(['Coffee_078', 'Coffee_079']))
    
    # Count
    included_sessions_by_mouse = sliced.groupby('mouse').size()
    

    ## Extract speed and smooth it
    # Results are somewhat sensitive to the degree of smoothing: more
    # smoothing means less time spent below the floor of stillness
    speed = this_synced_big_kinematics_df['dist_travelled'].copy() * 30
    speed = speed.rolling(window=6, center=True, min_periods=0).mean()
    
    # Split into center and sides
    speed_center = speed.loc[this_synced_big_kinematics_df['region'] == 8]
    speed_sides = speed.loc[this_synced_big_kinematics_df['region'] != 8]


    ## Set bins
    # Set floor for "stillness"
    # 1 px / frame is minimum detectable movement. Everything below that is 0.
    # This is 30 px / s, or approx 30 mm / s
    # Call anything below this threshold (in mm / s) stillness
    floor = 1 

    # Set bins
    bins = np.linspace(np.log10(floor), 3.5, 41)
    
    # Histogram each
    counts_sides, edges_sides = np.histogram(
        np.log10(speed_sides + floor),
        bins=bins)
    counts_center, edges_center = np.histogram(
        np.log10(speed_center + floor),
        bins=bins)

    # Normalize
    counts_center = counts_center / counts_center.sum()
    counts_sides = counts_sides / counts_sides.sum()
    
    
    ## Plot
    f, ax = my.plot.figure_1x1_small()
    my.plot.despine(ax)
    
    # Plot the actual counts
    # Skip the blank bins between floor and the lowest actual data point
    ax.plot(bins[6:-1], counts_center[6:], color='magenta')
    ax.plot(bins[6:-1], counts_sides[6:], color='green')
    
    # Plot stillness separately
    ax.plot(bins[0:1], counts_center[0:1], color='magenta', marker='o')
    ax.plot(bins[0:1], counts_sides[0:1], color='green', marker='o')
    
    # Pretty
    ax.set_xticks((0, 1, 2, 3))
    ax.set_xticklabels(('still', 1, 10, 100))
    ax.set_xlabel('speed (cm/s)')
    ax.set_ylabel('fraction of time')
    ax.set_ylim(ymin=0)
    ax.text(1, .2, 'center', color='magenta', ha='center')
    ax.text(1, .17, 'sides', color='green', ha='center')
    

    ## Save
    f.savefig('PLOT_HISTOGRAM_OF_SPEED_BY_REGION.svg')
    f.savefig('PLOT_HISTOGRAM_OF_SPEED_BY_REGION.png', dpi=300)


    ## Stats
    # Coffee_078 and Coffee_079 are included here although they never had HL,
    # not even sham
    with open('STATS__PLOT_HISTOGRAM_OF_SPEED_BY_REGION', 'w') as fi:
        fi.write('n = {} mice, {} sessions, {} trials, {} frames\n'.format(
            len(included_sessions_by_mouse),
            included_sessions_by_mouse.sum(),
            len(metrics_by_trial),
            len(synced_big_kinematics_df),
            ))
        fi.write('center: {:.3f} of frames below floor\n'.format(
            (speed_center < floor).mean()))
        fi.write('sides: {:.3f} of frames below floor\n'.format(
            (speed_sides < floor).mean()))
        fi.write('center: mean speed while moving {:.1f} cm / s\n'.format(
            speed_center[speed_center > floor].mean() / 10))
        fi.write('sides: mean speed while moving {:.1f} cm / s\n'.format(
            speed_sides[speed_sides > floor].mean() / 10))


## Helper function
def agg_by(df, col):
    agg = df.T.groupby(col).mean().T
    err = df.T.groupby(col).sem().T
    return agg, err


if QUANTIFY_CENTER_TRANSITS_BY_HL_TYPE:
    ## This plots speed and straightness of paths through the center, peri-HL
    ## The time range to use for learning plot and peri-HL plot
    # These are used inclusively
    from_start = (0, 15)
    from_hl = (-5, 10)


    ## Metrics to plot
    metric_l = [
        'speed_in_center',
        'straightness_in_center',
        ]

    name2pretty_name = {
        'speed_in_center': 'speed (cm/s)',
        'straightness_in_center': 'straightness',
        }
   
    
    ## Iterate over metrics
    # Keep stats on every metric here
    stats_l = []
    stats_keys_l = []
    
    # Iterate over metrics
    for metric in metric_l:
        ## Slice data
        # Choose one metric
        data_wrt_hl = metrics_wrt_hl[metric].unstack(
            'n_from_manipulation').T.copy()
        data_wrt_start = metrics_wrt_start[metric].unstack(
            'n_session').T.copy()
        
        # Convert to cm
        if metric == 'speed_in_center':
            data_wrt_hl = data_wrt_hl / 10
            data_wrt_start = data_wrt_start / 10
        
        # Keep a copy without interpolation
        data_wrt_hl_noint = data_wrt_hl.copy()
        data_wrt_start_noint = data_wrt_start.copy()
        
        # Interpolate the other copy
        data_wrt_hl = data_wrt_hl.interpolate(limit_area='inside')
        data_wrt_start = data_wrt_start.interpolate(limit_area='inside')
        
        # Slice by the time range we will use
        # This will be inclusive
        data_wrt_hl = data_wrt_hl.loc[slice(*from_hl)]
        data_wrt_start = data_wrt_start.loc[slice(*from_start)]
        data_wrt_hl_noint = data_wrt_hl_noint.loc[slice(*from_hl)]
        data_wrt_start_noint = data_wrt_start_noint.loc[slice(*from_start)]


        ## Aggregate over mice, using the non-interpolated data
        topl_group, topl_group_err = agg_by(data_wrt_hl_noint, 'manipulation2')
        
        
        ## Plot
        # Make handles
        f, ax = my.plot.figure_1x1_small()
        
        # Plot each group
        for group_name in topl_group.columns:
            # Color by group
            color = group2color[group_name]
            
            # Plot mean
            ax.plot(topl_group[group_name], color=color)
            
            # Shade error
            ax.fill_between(
                x=topl_group[group_name].index,
                y1=topl_group[group_name] - topl_group_err[group_name],
                y2=topl_group[group_name] + topl_group_err[group_name],
                color=color, alpha=.5, lw=0,
                )
        
        # Pretty
        my.plot.despine(ax)
        ax.set_xlim((-5, 10))
        ax.set_xticks((-5, 0, 5, 10))
        ax.set_xlabel('days after hearing loss')

        # ylim
        if metric == 'speed_in_center':
            ax.set_ylim((14, 35))
            ax.set_yticks((15, 25, 35))
        
        elif metric == 'straightness_in_center':
            ax.set_ylim((.55, .7))
            ax.set_yticks((.55, .6, .65, .7))

        # Label the axes accordingly
        pretty_name = name2pretty_name[metric]
        ax.set_ylabel(pretty_name)


        ## Run and store stats
        stats = extras.run_stats(data_wrt_start=None, data_wrt_hl=data_wrt_hl)
        stats_l.append(stats)
        stats_keys_l.append(metric)

    
        ## Savefig
        f.savefig(
            'QUANTIFY_CENTER_TRANSITS_BY_HL_TYPE_{}.svg'.format(metric)
            )
        f.savefig(
            'QUANTIFY_CENTER_TRANSITS_BY_HL_TYPE_{}.png'.format(metric),
            dpi=300)

    
        ## Stats
        n_sessions = metrics_wrt_hl.groupby(['manipulation2', 'mouse']).size()
        n_mice = n_sessions.groupby('manipulation2').size()
        
        # Write to file
        stats_filename = 'STATS__QUANTIFY_CENTER_TRANSITS_BY_HL_TYPE_{}'.format(
            metric)
        with open(stats_filename, 'w') as fi:
            # Write N
            fi.write('N mice per group:\n{}\n\ntotal mice: {}\n'.format(
                n_mice,
                n_mice.sum(),
                ))       
            
            # Write p-values for t-test before/after HL
            fi.write('t-test on bilateral before vs after: {:.4f}\n'.format(
                stats['peri_HL__hl_bilateral__pval']))
            fi.write('t-test on unilateral before vs after: {:.4f}\n'.format(
                stats['peri_HL__hl_unilateral__pval']))
            fi.write('t-test on sham before vs after: {:.4f}\n'.format(
                stats['peri_HL__hl_sham__pval']))

            # Write inter-group p-values after HL
            # For some reason the order of these levels in the formed string
            # swapped at some point
            fi.write('p-value on bilateral vs unilateral after HL: {:.4f}\n'.format(
                stats['hl_recovery__inter_group__hl_unilateral_VS_hl_bilateral__pval']))
            fi.write('p-value on bilateral vs sham after HL: {:.4f}\n'.format(
                stats['hl_recovery__inter_group__hl_sham_VS_hl_bilateral__pval']))
            fi.write('p-value on sham vs unilateral after HL: {:.4f}\n'.format(
                stats['hl_recovery__inter_group__hl_unilateral_VS_hl_sham__pval']))


if QUANTIFY_ENTRIES_THROUGHOUT_LEARNING_AND_HL:
    ## Plots entries of various types during learning and peri-HL
    # The same type of plot is made for many different metrics
    # And runs stats on each
    
    ## The time range to use for learning plot and peri-HL plot
    # These are used inclusively
    from_start = (0, 15)
    from_hl = (-5, 10)


    ## Which metrics to plot
    metric_l = [
        'is_dup', 
        #~ 'into_prp', 
        #~ 'into_prp_consumm',
        'frac_returned',
        'ducks + consumms', 
        'consumms', 
        'ducks', 
        'p_cycling',
        ]

    name2pretty_name = {
        'ducks': 'entries without poke',
        'is_dup': 'duplicate entries',
        'consumms': 'entries with poke',
        'ducks + consumms': 'total entries',
        'p_cycling': 'probability of cycling',
        'into_prp': 'entry into\npreviously rewarded port',
        'frac_returned': 'probability of\nreturning to previously\nrewarded chamber',
        }
   
    
    ## Iterate over metrics
    # Keep stats on every metric here
    stats_l = []
    stats_keys_l = []
    
    # Iterate over metrics
    for metric in metric_l:
        ## Slice data
        # Choose one metric
        data_wrt_hl = metrics_wrt_hl[metric].unstack(
            'n_from_manipulation').T.copy()
        data_wrt_start = metrics_wrt_start[metric].unstack(
            'n_session').T.copy()
        
        # Keep a copy without interpolation
        data_wrt_hl_noint = data_wrt_hl.copy()
        data_wrt_start_noint = data_wrt_start.copy()
        
        # Interpolate the other copy
        data_wrt_hl = data_wrt_hl.interpolate(limit_area='inside')
        data_wrt_start = data_wrt_start.interpolate(limit_area='inside')
        
        # Slice by the time range we will use
        # This will be inclusive
        data_wrt_hl = data_wrt_hl.loc[slice(*from_hl)]
        data_wrt_start = data_wrt_start.loc[slice(*from_start)]
        data_wrt_hl_noint = data_wrt_hl_noint.loc[slice(*from_hl)]
        data_wrt_start_noint = data_wrt_start_noint.loc[slice(*from_start)]


        ## Aggregate over mice, using the non-interpolated data
        topl_group_wrt_hl, topl_group_wrt_hl_err = agg_by(
            data_wrt_hl_noint, 'manipulation2')
        topl_group_wrt_start, topl_group_wrt_start_err = agg_by(
            data_wrt_start_noint, 'manipulation2')


        ## Figure handles
        f, axa = plt.subplots(1, 2, figsize=(2.75, 2), sharey=True)
        f.subplots_adjust(left=.2, right=.9, wspace=.1, bottom=.225, top=.9)
        
       
        ## Plot wrt start
        # Choose ax
        ax = axa[0]
        extras.make_plot(
            ax, topl_group_wrt_start, topl_group_wrt_start_err, metric, 
            group2color)

        # Pretty
        my.plot.despine(ax)    
        ax.set_xticks((0, 10))
        ax.set_xticklabels(('start', 'start+10'))
        
        
        ## Plot wrt hl
        # Choose ax
        ax = axa[1]
        extras.make_plot(
            ax, topl_group_wrt_hl, topl_group_wrt_hl_err, metric, group2color)

        # Pretty
        my.plot.despine(ax)    
        my.plot.despine(ax, which=('left',))
        ax.set_xticks((0, 10))
        ax.set_xticklabels(('HL', 'HL+10'))

        
        ## Put shared title into first ax
        # Get pretty name if there is one
        try:
            pretty_name = name2pretty_name[metric]
        except KeyError:
            pretty_name = metric
        
        # Set ylabel in the first ax
        ax = axa[0]
        ax.set_ylabel(pretty_name)
    
    
        ## Run and store stats
        stats = extras.run_stats(
            data_wrt_start=data_wrt_start, data_wrt_hl=data_wrt_hl)
        stats_l.append(stats)
        stats_keys_l.append(metric)

    
        ## Save
        f.savefig(
            'QUANTIFY_ENTRIES_THROUGHOUT_LEARNING_AND_HL_{}.svg'.format(
            metric))
        f.savefig(
            'QUANTIFY_ENTRIES_THROUGHOUT_LEARNING_AND_HL_{}.png'.format(
            metric), dpi=300)        
    
    
    ## Concat stats
    stats_df = pandas.DataFrame(stats_l, index=stats_keys_l).T
    stats_df.index.name = 'stat'
    stats_df.columns.name = 'metric'
    
    # Write them out
    extras.write_out_stats(stats_df)

    
plt.show()
