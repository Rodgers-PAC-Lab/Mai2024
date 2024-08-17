## This directory generates behavior figures that don't require video
# main2a is for pre-HL learning curves by different groups of mice

import numpy as np
import os
import pandas
import datetime
import matplotlib.pyplot as plt
import my.plot
import scipy.stats


## Setup
# Figures
my.plot.manuscript_defaults()
my.plot.font_embed()

# Paths
repository_dir = os.path.expanduser(
    '~/mnt/cuttlefish/chris/data/20240817_ss_paper')
    
    
## Load behavior data
# Note that this includes some mice that were trained on SweepTask
# and VariabilityTask from the get-go
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


## Drop post-manipulation sessions because this script is only pre-manipulation
perf_metrics = perf_metrics[
    perf_metrics['n_from_manipulation'].isnull() | 
    (perf_metrics['n_from_manipulation'] < 0)
    ]

# Apply this same drop to trial_data
trial_data = my.misc.slice_df_by_some_levels(
    trial_data, pandas.Index(perf_metrics['session_name']))


## Calculate learning time in trials as well as sessions
# Join session-wise learning metrics onto trial_data
# Join "n_session" (which defined "learning_time" in main1c) from 
# perf_metrics to trial_data
trial_data = trial_data.join(
    perf_metrics.reset_index().set_index(
    ['mouse', 'session_name'])['n_session'])
assert not trial_data['n_session'].isnull().any()

# Join "learning time"
trial_data = trial_data.join(mouse_data['learning_time'])
assert not trial_data['learning_time'].isnull().any()

# Calculate learning time in trials 
# Total number of trials, up to an including the end of the session specified
# by "learning_time", which is the first session whose performance passed
# performance threshold (see main1c)
mouse_data['learning_time_in_trials'] = trial_data.loc[
    trial_data['n_session'] <= trial_data['learning_time']
    ].groupby('mouse').size()


## Smooth the perf_metrics over trials instead of sessions
TRIAL_SMOOTHING_WIN_STD = 50
for mouse, sub_tm in trial_data.groupby('mouse'):
    # Smooth
    sub_tm['rcp_smooth'] = sub_tm['rcp'].rolling(
        window=5*TRIAL_SMOOTHING_WIN_STD, center=True, min_periods=1, 
        win_type='gaussian').mean(
        std=TRIAL_SMOOTHING_WIN_STD)
    
    # Store
    trial_data.loc[sub_tm.index, 'rcp_smooth'] = sub_tm['rcp_smooth']

    # Smooth
    sub_tm['fc_smooth'] = (sub_tm['outcome'] == 'correct').astype(int).rolling(
        window=5*TRIAL_SMOOTHING_WIN_STD, center=True, min_periods=1, 
        win_type='gaussian').mean(
        std=TRIAL_SMOOTHING_WIN_STD)
    
    # Store
    trial_data.loc[sub_tm.index, 'fc_smooth'] = sub_tm['fc_smooth']

    # Also store total trials so far
    trial_data.loc[sub_tm.index, 'n_trials_total'] = range(len(sub_tm))


## Recalculate perf metrics, now including non-consummatory PRP
# TODO: move to another script
# This is copied from the parsing code
# Calculate the rank of each poke, excluding ALL prp
# Get the latency to each port on each trial, excluding consummatory only
# but retaining the other PRP
this_poke_df = poke_data.loc[~poke_data['consummatory']]
latency_by_port = this_poke_df.reset_index().groupby(
    ['mouse', 'session_name', 'trial', 'poked_port'])['t_wrt_start'].min()

# Unstack the port onto columns
lbpd_unstacked = latency_by_port.unstack('poked_port')

# Rank them in order of poking
# Subtract 1 because it starts with 1
# After future_stack, we now dropna and sort
lbpd_ranked = lbpd_unstacked.rank(
    method='first', axis=1).stack(
    future_stack=True).dropna().sort_index().astype(int) - 1

# Exception: RCP can be 7 on the first trial of the session, because no PRP
# No longer true, RCP can now be 7
#assert lbpd_ranked.drop(0, level='trial').max() <= 6

# Join this rank onto poke_data
poke_data = poke_data.join(lbpd_ranked.rename('poke_rank_wprp'), 
    on=['mouse', 'session_name', 'trial', 'poked_port'])

# Insert poke_rank of -1 wherever consummatory
assert (
    poke_data.loc[poke_data['poke_rank'].isnull(), 'consummatory'] 
    == True).all()
poke_data.loc[poke_data['consummatory'], 'poke_rank_wprp'] = -1
poke_data['poke_rank_wprp'] = poke_data['poke_rank_wprp'].astype(int)

# Calculate rank of correct poke
correct_port = trial_data['rewarded_port'].dropna().reset_index()
cp_idx = pandas.MultiIndex.from_frame(correct_port)
rank_of_correct_port = lbpd_ranked.reindex(
    cp_idx).droplevel('rewarded_port').rename('rcp_wprp')

# Append this to trial_data
trial_data = trial_data.join(rank_of_correct_port)

# Error check: rcp is null iff no_correct_pokes
assert trial_data.loc[
    trial_data['rcp_wprp'].isnull()]['no_correct_pokes'].all()
assert trial_data.loc[
    trial_data['no_correct_pokes'], 'rcp_wprp'].isnull().all()

# Calculate frac_return (those where they returned to PRP)
trial_data['prp_returned'] = trial_data['rcp'] != trial_data['rcp_wprp']

# Join rcp_wprp on perf_metrics
perf_metrics = perf_metrics.join(
    trial_data.groupby('session_name')[['rcp_wprp', 'prp_returned']].mean(), 
    on='session_name')

# The join loses the columns' name
perf_metrics.columns.name = 'metric'


## Interpolate perf_metrics over missing sessions (those with too few trials)
# Calculate the last n_session for each mouse
n_sessions_by_mouse = perf_metrics.groupby('mouse').size()

# Calculate which sessions are "missing" (because the mice learned quickly
#  and we stopped training them). Use this to set the x-axis on learning curves
# At session 19 (the 20th session), 24.6% are missing
# At session 20 27.5% are missing
# At session 23 30.4% are missing
# Thus, it seems reasonable to show only the first 20 sessions or so
frac_missing_sessions = (
    perf_metrics['n_trials'].unstack('mouse').isnull().mean(1))

# Blank out data when n_trials too few
# n_trials mode is about 85, but goes all the way down to 0
# no clear way to choose the threshold for blanking
perf_metrics['too_few_trials'] = perf_metrics['n_trials'] < 10
perf_metrics.loc[perf_metrics['too_few_trials'], 'handedness'] = np.nan
perf_metrics.loc[perf_metrics['too_few_trials'], 'fc'] = np.nan
perf_metrics.loc[perf_metrics['too_few_trials'], 'rcp'] = np.nan

# all analysis is done on this blanked data
# use this "non-interpolated" version for statistics
# or should we do this on the non-blanked data?
unstacked_pm_noint = perf_metrics[
    ['handedness', 'n_trials', 'fc', 'rcp', 'rcp_wprp', 'prp_returned']
    ].unstack('mouse')

# for visualization only, interpolate through those low-trial days
unstacked_pm = unstacked_pm_noint.interpolate(limit_area='inside')

# this version is just for learning over trials instead of sessions, and
# it includes all trials even those from low-trial sessions
unstacked_pm_by_trial = trial_data.reset_index()[
    ['mouse', 'n_trials_total', 'fc_smooth', 'rcp_smooth']].set_index(
    ['mouse', 'n_trials_total']).unstack('mouse')


## Plots
PRINT_OUT_ALL_PROTOCOLS = True
PRINT_MOUSE_TABLE = True
PLOT_LEARNING_CURVE_BY_SEX = True
PLOT_BY_COHORT = True
PLOT_BY_SWEEP_VS_FIXED = True
PLOT_LEARNING_CURVE_ALL_MICE = True
PLOT_LEARNING_CURVE_WPRP = True
PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL = True
HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS = True


## Helper function to do plot and stats consistently
def plot_perf_metrics(gobj, group2color, xmax=30):
    ## Figure handles
    f, axa = plt.subplots(1, 3, figsize=(9, 2.1), sharex=True)
    f.subplots_adjust(top=.9, left=.08, right=.8, wspace=.7, bottom=.275)

    
    ## Put the provided groups on the columns of unstacked_pm and unstacked_pm_noint
    # Turn gobj.groups into a DataFrame
    series_l = []
    series_keys_l = []
    for group_label, group_idx in gobj.groups.items():
        series_l.append(pandas.Series(group_idx, name='mouse'))
        series_keys_l.append(group_label)
    grouping_idx = pandas.concat(
        series_l, keys=series_keys_l, names=['group', 'idx'])
    grouping_idx = grouping_idx.reset_index().drop('idx', axis=1).set_index(
        'mouse')['group']
    
    # Join
    this_unstacked_pm = my.misc.join_level_onto_index(
        unstacked_pm.T, grouping_idx).T
    this_unstacked_pm_noint = my.misc.join_level_onto_index(
        unstacked_pm_noint.T, grouping_idx).T
    
    # Drop any null groups
    this_unstacked_pm = this_unstacked_pm.drop(
        np.nan, axis=1, level='group')
    this_unstacked_pm_noint = this_unstacked_pm_noint.drop(
        np.nan, axis=1, level='group')
    
    
    ## Iterate over groups
    for group_label in this_unstacked_pm.columns.levels[0]:
        # Get color for this group
        color = group2color[group_label]

        # Plot fc
        ax = axa[0]
        topl = this_unstacked_pm_noint.loc[:, group_label].loc[:, 'fc']
        ax.plot(topl.mean(1), color=color, lw=1.5)
        ax.fill_between(
            x=topl.index,
            y1=topl.mean(1) - topl.std(1),
            y2=topl.mean(1) + topl.std(1),
            color=color, lw=0, alpha=.25)
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .5, 1))
        my.plot.despine(ax)
        ax.set_ylabel('fraction correct')
        
        # Plot rcp
        ax = axa[1]
        topl = 1 + this_unstacked_pm_noint.loc[:, group_label].loc[:, 'rcp']
        ax.plot(topl.mean(1), color=color, lw=1.5)
        ax.fill_between(
            x=topl.index,
            y1=topl.mean(1) - topl.std(1),
            y2=topl.mean(1) + topl.std(1),
            color=color, lw=0, alpha=.25)
        ax.set_ylim((1, 4.75))
        ax.set_yticks((1, 2, 3, 4))
        my.plot.despine(ax)
        ax.set_ylabel('poked ports per trial')

        # Plot n_trials
        # This is the only one to use the interpolated data for, because
        # individual mice are plotted. All others should use uninterpolated
        ax = axa[2]
        topl = this_unstacked_pm.loc[:, group_label].loc[:, 'n_trials']
        ax.plot(topl.mean(1), color=color, lw=1.5)
        ax.fill_between(
            x=topl.index,
            y1=topl.mean(1) - topl.std(1),
            y2=topl.mean(1) + topl.std(1),
            color=color, lw=0, alpha=.25)
        ax.set_ylim(ymin=0, ymax=170)
        ax.set_yticks((0, 50, 100, 150))
        my.plot.despine(ax)
        ax.set_ylabel('number of trials')


    ## Pretty
    # Cut off at 30 session, because that's when the last mouse learns
    axa[0].set_xticks((0, 15, 30))
    axa[0].set_xlim((0, xmax))

    # Chance line
    axa[0].plot([0, xmax], [1/7, 1/7], 'k--', lw=1)
    axa[1].plot([0, xmax], [4, 4], 'k--', lw=1)

    # x-axis label
    axa[0].set_xlabel('session number')
    axa[1].set_xlabel('session number')
    axa[2].set_xlabel('session number')

    
    ## Stats
    metric_l = ['n_trials', 'fc', 'rcp']
    stats_df_l = []
    pval_l = []
    keys_l = []
    for metric in metric_l:
        # Extract data for this metric
        stat_data = this_unstacked_pm_noint.xs(metric, axis=1, level='metric')

        # Cut off at the last plotted session (inclusive, like xlim)
        stat_data = stat_data.loc[:30]

        # Stack into tall
        aov_data = stat_data.stack(
            future_stack=True).stack(future_stack=True).dropna().rename(
            metric).reset_index()

        # run AOV
        aov_res = my.stats.anova(
            aov_data, '{} ~ mouse + n_session + group'.format(metric))
        
        # parse
        group_pval = aov_res['pvals']['p_group']
        
        # For effect size, just use the marginal, I don't know how to 
        # interpret the fit given that each mouse also has its own fit
        # (Zero-meaning each mouse and then removing mouse from the AOV
        # yields a fit that matches this calculation)
        group_mean = aov_data.groupby('group')[metric].mean().rename('mean')
        group_n_sessions = aov_data.groupby('group')[metric].size().rename('n_sessions')
        group_n_mice = aov_data.groupby(['group', 'mouse']).size().groupby('group').size().rename('n_mice')
        
        # concat into stats_df
        stats_df = pandas.concat([group_mean, group_n_sessions, group_n_mice], axis=1)
        
        # double-check that the N matches the plots above
        for group in gobj.groups.keys():
            assert (
                stats_df.loc[group, 'n_mice'] == 
                this_unstacked_pm.loc[:, group].loc[:, metric].shape[1]
                )
        
        # Store
        pval_l.append(group_pval)
        stats_df_l.append(stats_df)
        keys_l.append(metric)
    
    # Concat over metrics
    big_stats_df = pandas.concat(stats_df_l, keys=keys_l, names=['metric'])
    pvalues_ser = pandas.Series(pval_l, index=pandas.Index(keys_l, name='metric'))
    
    
    # return stats
    return f, axa, big_stats_df, pvalues_ser


if PRINT_OUT_ALL_PROTOCOLS:
    # Print out protocol used for old cohorts
    # Supplemental material
    # For notes on how the protocols differ, see 20221030_apan_poster/main1.py
    # and 20221030_apan_poster/protocols and autopilot/protocols
    protocols_used = session_data.join(mouse_data['cohort']).groupby(
        ['cohort', 'protocol_name']).size()
    
    print(protocols_used)
    
    # "VN" matches the variant number in the supplemental material
    # Cohort 1 Summer 2022: 
    #  V1. ControlTest220525 - Same as fixed_new except variability 1 ms instead of 31 ms (higher reward)
    #  V1. ControlTest220614 - Same as fixed_new except variability 1 ms instead of 31 ms (lower reward)
    #  V2. Sweep220525 - Same as variable_new except 
    #    rate is 1-12 Hz instead of 3-10 Hz and 
    #    target is 5-18 kHz instead of 5-15 kHz and 
    #    level is 50-90 instead of 65-90 (ie amplitudes -3 to -1)
    #  Sweep220617 - Same as variable_new
    #  V3. Sweep220629 - Same as variable_new except
    #    rate is 10 Hz fixed, variability is 10 ms fixed, level is 80 dB fixed, and
    #    bandwidth varies from 3 kHz to 9 kHz, 
    #  V4. VariabilityTest220525 - Same as fixed_new except variability ranges from 1 ms to 316 ms
    # Cohort 2 Fall 2022: 
    #  V1. ControlTest220614 - see above
    #  V5. ControlTest220930 - same as other ControlTest except 6 Hz rate and amplitude -1.5
    # Cohort 3 Winter 2022: 
    #  V5. ControlTest220930 - see above
    # 
    # To convert amplitudes to dB SPL, use 20 * -2 + 110 = 70 dB, 
    # replacing -2 with desired amplitude


## Print out a table of the mice that were used and their learning times
if PRINT_MOUSE_TABLE:
    mouse_data2 = mouse_data.copy()

    # Map the various fixed and variable protocols into standard categories
    mouse_data2['protocol_mostly_used'] = mouse_data2['protocol_mostly_used'].replace({
        'ControlTest220930': 'fixed_old',
        'ControlTest220525': 'fixed_old',
        'ControlTest220614': 'fixed_old',
        'VariabilityTest220525': 'variable_old',
        'Sweep220629': 'variable_old',
        'Sweep220525': 'variable_old',
        '230201_SSS_sweep': 'variable_new',
        '230201_SSS_fixed': 'fixed_new',
        })

    # Map the various hearing loss manipulations into standard categories
    mouse_data2['manipulation'] = mouse_data2['manipulation'].replace({
        'hl_bilateral': 'hl',
        'hl_left': 'hl',
        'hl_right': 'hl',
        'lesion_AC': 'no hl',
        'lesion_VC': 'no hl',
        'hl_sham': 'hl', # only for this table do we count hl_sham as hl
        'none': 'no hl',
        })

    # group into a table
    table = mouse_data2.groupby(
        ['cohort', 'sex', 'manipulation', 'protocol_mostly_used']).size().unstack(
        'sex').fillna(0).astype(int).reindex(
        ['summer 2022', 'fall 2022', 'winter 2022', 'spring 2023', 'summer 2023'], 
        level=0)
    table['N'] = table.sum(1)

    # Dump the table
    with open('STATS__TABLE_OF_MOUSE_N', 'w') as fi:
        fi.write(str(table) + '\n\n')
        
        fi.write('mouse age 25-50-75 quartiles\n')
        fi.write(str(mouse_data['start_age'].quantile([.25, .5, .75])) + '\n\n')
        fi.write('grand total: {} mice\n'.format(table['N'].sum()))

    # Print the table
    with open('STATS__TABLE_OF_MOUSE_N') as fi:
        print(''.join(fi.readlines()))


## Plot by sex
if PLOT_LEARNING_CURVE_BY_SEX:
    
    ## Group by sex
    gobj = mouse_data.groupby('sex')
    n_each_group = gobj.size()
    group2color = {
        'M': 'blue',
        'F': 'red',
        }
    group2pretty_name = {
        'M': 'male',
        'F': 'female',
        }
    
    
    ## Plot and get stats
    f, axa, stats_df, pvalues_ser = plot_perf_metrics(gobj, group2color, xmax=30)

    # Calculate an effect size
    effect_size = stats_df['mean'].unstack()    
    effect_size['diff'] = effect_size.max(axis=1) - effect_size.min(axis=1)
    effect_size['diff/mean'] = effect_size['diff'] / effect_size.mean(axis=1)
    

    ## Pretty
    # Legend
    for n_group, (group, color) in enumerate(group2color.items()):
        pretty_name = group2pretty_name[group]
        y = .95 - .08 * n_group
        group_size = len(gobj.groups[group])
        
        f.text(
            .92, y, '{} (n = {})'.format(pretty_name, group_size), 
            color=color, ha='center', va='center')
    

    ## Save
    f.savefig('PLOT_LEARNING_CURVE_BY_SEX.svg')
    f.savefig('PLOT_LEARNING_CURVE_BY_SEX.png', dpi=300)
    
    
    ## Stats
    # Trial count for males vs females now slightly differs from paper
    # because post-lesion sessions excluded here
    with open('STATS__PLOT_LEARNING_CURVE_BY_SEX', 'w') as fi:
        fi.write(str(stats_df) + '\n')
        fi.write('\npvalues for effect of group:\n')
        fi.write(str(pvalues_ser) + '\n')
        fi.write('\neffect size:\n')
        fi.write(str(effect_size) + '\n')
        fi.write('\nmales perform {:.2g}\% more trials than females\n'.format(
            100 * (effect_size.loc['n_trials', 'M'] / 
            effect_size.loc['n_trials', 'F'] - 1)))

    # Print back
    with open('STATS__PLOT_LEARNING_CURVE_BY_SEX') as fi:
        print('STATS__PLOT_LEARNING_CURVE_BY_SEX\n')
        print(''.join(fi.readlines()))    


if PLOT_BY_COHORT:
    ## Group by cohort
    gobj = mouse_data.groupby('cohort')
    n_each_group = gobj.size()
    group2color = {
        'summer 2022': 'red', 
        'fall 2022': 'orange', 
        'winter 2022': 'green',
        'spring 2023': 'blue', 
        'summer 2023': 'purple', 
        }
    group2pretty_name = {
        'summer 2022': 'cohort 1', 
        'fall 2022': 'cohort 2', 
        'winter 2022': 'cohort 3',
        'spring 2023': 'cohort 4', 
        'summer 2023': 'cohort 5', 
        }
    
    
    ## Plot and get stats
    f, axa, stats_df, pvalues_ser = plot_perf_metrics(gobj, group2color, xmax=30)

    # Calculate an effect size
    effect_size = stats_df['mean'].unstack()    
    effect_size['diff'] = effect_size.max(axis=1) - effect_size.min(axis=1)
    effect_size['diff/mean'] = effect_size['diff'] / effect_size.mean(axis=1)
    

    ## Pretty
    # Legend
    for n_group, (group, color) in enumerate(group2color.items()):
        pretty_name = group2pretty_name[group]
        y = .95 - .08 * n_group
        group_size = len(gobj.groups[group])
        
        f.text(
            .83, y, '{} (n = {})'.format(pretty_name, group_size), 
            color=color, ha='left', va='center')
    

    ## Save
    f.savefig('PLOT_BY_COHORT.svg')
    f.savefig('PLOT_BY_COHORT.png', dpi=300)


    ## Stats
    with open('STATS__PLOT_BY_COHORT', 'w') as fi:
        fi.write(str(stats_df) + '\n')
        fi.write('\npvalues for effect of group:\n')
        fi.write(str(pvalues_ser) + '\n')
        fi.write('\neffect size:\n')
        fi.write(str(effect_size) + '\n')

    # Print back
    with open('STATS__PLOT_BY_COHORT') as fi:
        print('STATS__PLOT_BY_COHORT')
        print(''.join(fi.readlines()))    


if PLOT_BY_SWEEP_VS_FIXED:
    ## Group by protocol_mostly_used
    gobj = mouse_data[
        mouse_data['protocol_mostly_used'].isin(
        ['230201_SSS_fixed', '230201_SSS_sweep']
        )].groupby('protocol_mostly_used')
    
    n_each_group = gobj.size()
    group2color = {
        '230201_SSS_fixed': 'magenta', 
        '230201_SSS_sweep': 'green', 
        }
    group2pretty_name = {
        '230201_SSS_fixed': 'fixed', 
        '230201_SSS_sweep': 'variable', 
        }
    
    
    ## Plot and get stats
    f, axa, stats_df, pvalues_ser = plot_perf_metrics(gobj, group2color, xmax=30)

    # Calculate an effect size
    effect_size = stats_df['mean'].unstack()    
    effect_size['diff'] = effect_size.max(axis=1) - effect_size.min(axis=1)
    effect_size['diff/mean'] = effect_size['diff'] / effect_size.mean(axis=1)
    

    ## Pretty
    # Legend
    for n_group, (group, color) in enumerate(group2color.items()):
        pretty_name = group2pretty_name[group]
        y = .95 - .08 * n_group
        group_size = len(gobj.groups[group])
        
        f.text(
            .92, y, '{} (n = {})'.format(pretty_name, group_size), 
            color=color, ha='center', va='center')
    

    ## Save
    f.savefig('PLOT_BY_SWEEP_VS_FIXED.svg')
    f.savefig('PLOT_BY_SWEEP_VS_FIXED.png', dpi=300)


    ## Stats
    with open('STATS__PLOT_BY_SWEEP_VS_FIXED', 'w') as fi:
        fi.write(str(stats_df) + '\n')
        fi.write('\npvalues for effect of group:\n')
        fi.write(str(pvalues_ser) + '\n')
        fi.write('\neffect size:\n')
        fi.write(str(effect_size) + '\n')

    # Print back
    with open('STATS__PLOT_BY_SWEEP_VS_FIXED') as fi:
        print('STATS__PLOT_BY_SWEEP_VS_FIXED')
        print(''.join(fi.readlines()))    


## Plot learning curve for all
if PLOT_LEARNING_CURVE_ALL_MICE:
    ## Calculate criterion day
    res_l = []
    topl = 1 + unstacked_pm['rcp']
    for mouse_name in topl.columns:
        learning_day = mouse_data.loc[mouse_name, 'learning_time']
        
        # TODO: fix the issue where this is sometimes null
        # This is from some mice that got put on another task after only
        # 10 days
        if pandas.isnull(learning_day):
            print("warning: {} never learned".format(mouse_name))
            continue
        
        perf_that_day = topl.loc[learning_day, mouse_name]
        res_l.append((learning_day, perf_that_day))
    

    ## Figure handles
    f, axa = plt.subplots(1, 3, figsize=(9, 2.1), sharex=True)
    f.subplots_adjust(top=.9, left=.08, right=.8, wspace=.7, bottom=.275)

    # Plot fc
    ax = axa[0]
    topl = unstacked_pm['fc']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.plot(ax.get_xlim(), [1/7, 1/7], 'k--', lw=1)
    my.plot.despine(ax)
    ax.set_ylabel('fraction correct')

    # Plot rcp
    ax = axa[1]
    topl = 1 + unstacked_pm['rcp']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim((1, 4.75))
    ax.set_yticks((1, 2, 3, 4))
    ax.plot(ax.get_xlim(), [4, 4], 'k--', lw=1)
    my.plot.despine(ax)
    ax.set_ylabel('ports poked per trial')

    # Plot the dots
    axa[1].plot(
        np.array(res_l)[:, 0], np.array(res_l)[:, 1], 
        marker='.', color='r', alpha=1, ls='none', ms=3)

    # Plot n_trials
    ax = axa[2]
    topl = unstacked_pm['n_trials']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim(ymin=0)
    my.plot.despine(ax)
    ax.set_ylabel('number of trials')
    
    # Cut off at 32 session, because that's when the last mouse learns
    ax.set_xlim((0, 32))
    ax.set_xticks((0, 15, 30))

    # Shared x-axis
    axa[0].set_xlabel('session number')
    axa[1].set_xlabel('session number')
    axa[2].set_xlabel('session number')

    # Count mice
    n_mice = len(unstacked_pm.columns.get_level_values(1).unique())

    # Legend
    f.text(
        .92, .9, 'n = {} mice'.format(n_mice), 
        color='k', ha='center', va='center')
    
    # Save
    f.savefig('PLOT_LEARNING_CURVE_ALL_MICE.svg')
    f.savefig('PLOT_LEARNING_CURVE_ALL_MICE.png', dpi=300)

    # Write stats
    with open('STATS__PLOT_LEARNING_CURVE_ALL_MICE', 'w') as fi:
        ltq = mouse_data['learning_time'].quantile((.25, .5, .75))
        fi.write('learning time quantiles:\n')
        fi.write(str(ltq) + '\n')

    # Print back
    with open('STATS__PLOT_LEARNING_CURVE_ALL_MICE') as fi:
        print('STATS__PLOT_LEARNING_CURVE_ALL_MICE\n')
        print(''.join(fi.readlines()))    


if PLOT_LEARNING_CURVE_WPRP:
    ## Include the trials where the mouse returned to the PRP
    
    ## Figure handles
    f, axa = my.plot.figure_1x2_standard(sharex=True)
    #f.subplots_adjust(top=.9, left=.08, right=.8, wspace=.7, bottom=.275)
    f.subplots_adjust(wspace=.6, right=.98)

    # Plot prp_returned
    ax = axa[0]
    topl = unstacked_pm['prp_returned']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    my.plot.despine(ax)
    ax.set_ylabel('probability of returning to\npreviously rewarded port')

    # Plot rcp
    ax = axa[1]
    topl = 1 + unstacked_pm['rcp']
    ax.plot(topl.mean(1), color='magenta')
    ax.fill_between(
        x=topl.index,
        y1=topl.mean(1) - topl.sem(1),
        y2=topl.mean(1) + topl.sem(1),
        color='magenta',
        alpha=.5,
        lw=0,
        )
    topl = 1 + unstacked_pm['rcp_wprp']
    ax.plot(topl.mean(1), '-', color='green')
    ax.fill_between(
        x=topl.index,
        y1=topl.mean(1) - topl.sem(1),
        y2=topl.mean(1) + topl.sem(1),
        color='green',
        alpha=.5,
        lw=0,
        )    
    ax.set_ylim((1, 4.75))
    ax.set_yticks((1, 2, 3, 4))
    #ax.plot(ax.get_xlim(), [4, 4], 'k--', lw=1)
    my.plot.despine(ax)
    ax.set_ylabel('ports poked per trial\n(lower is better)')

    # Cut off at 32 session, because that's when the last mouse learns
    ax.set_xlim((0, 32))
    ax.set_xticks((0, 15, 30))

    # Shared x-axis
    axa[0].set_xlabel('session number')
    axa[1].set_xlabel('session number')

    # Count mice
    n_mice = len(unstacked_pm.columns.get_level_values(1).unique())

    # Legend
    f.text(
        .9, .85, 'n = {} mice'.format(n_mice), 
        color='k', ha='center', va='center')
    
    # Save
    f.savefig('PLOT_LEARNING_CURVE_WPRP.svg')
    f.savefig('PLOT_LEARNING_CURVE_WPRP.png', dpi=300)


## Plot learning curve for all
if PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL:
    ## Figure handles
    f, axa = plt.subplots(1, 3, figsize=(9, 2.1), sharex=True)
    f.subplots_adjust(top=.9, left=.08, right=.8, wspace=.7, bottom=.275)

    # Nothing to plot in axa 2
    axa[2].set_visible(False)

    # Plot fc
    ax = axa[0]
    topl = unstacked_pm_by_trial['fc_smooth']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.plot(ax.get_xlim(), [1/7, 1/7], 'k--', lw=1)
    my.plot.despine(ax)
    ax.set_ylabel('fraction correct')

    # Plot rcp
    ax = axa[1]
    topl = 1 + unstacked_pm_by_trial['rcp_smooth']
    ax.plot(topl, color='gray', lw=.5, alpha=.5)
    ax.plot(topl.mean(1), 'k-')
    ax.set_ylim((1, 4.75))
    ax.set_yticks((1, 2, 3, 4))
    ax.plot(ax.get_xlim(), [4, 4], 'k--', lw=1)
    my.plot.despine(ax)
    ax.set_ylabel('ports poked per trial')

    # Red dot on rcp where learning occurred
    res_l = []
    for mouse_name in topl.columns:
        learning_day = mouse_data.loc[mouse_name, 'learning_time_in_trials']
        
        if pandas.isnull(learning_day):
            print("warning: {} never learned".format(mouse_name))
            continue
        
        perf_that_day = topl.loc[learning_day, mouse_name]
        res_l.append((learning_day, perf_that_day))
    
    # Plot the dots
    ax.plot(
        np.array(res_l)[:, 0], np.array(res_l)[:, 1], 
        marker='.', color='r', alpha=1, ls='none', ms=3)

    # Cut off at 2500 trial, because that's when the last mouse learns
    ax.set_xlim((0, 2500))
    ax.set_xticks((0, 1250, 2500))

    # Shared x-axis
    axa[1].set_xlabel('cumulative trial number')
    axa[2].set_xlabel('cumulative trial number')

    # Count mice
    n_mice = len(unstacked_pm.columns.get_level_values(1).unique())

    # Legend
    f.text(
        .92, .9, 'n = {} mice'.format(n_mice), 
        color='k', ha='center', va='center')
    
    # Save
    f.savefig('PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL.svg')
    f.savefig('PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL.png', dpi=300)

    # Write stats
    with open('STATS__PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL', 'w') as fi:
        ltq = mouse_data['learning_time_in_trials'].quantile((.25, .5, .75))
        fi.write('learning time quantiles:\n')
        fi.write(str(ltq) + '\n')

    # Print back
    with open('STATS__PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL') as fi:
        print('STATS__PLOT_LEARNING_CURVE_ALL_MICE_BY_TRIAL\n')
        print(''.join(fi.readlines()))    


## Hist the learning time in sessions and trials
if HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS:
    f, axa = plt.subplots(1, 2, figsize=(4, 2.1), sharey=True)
    f.subplots_adjust(top=.9, left=.15, right=.95, wspace=.3, bottom=.275)

    axa[0].hist(mouse_data['learning_time'], bins=15, histtype='step', color='k')
    axa[0].set_xlabel('sessions')
    axa[0].set_xlim((0, 32))
    axa[0].set_xticks((0, 15, 30))
    axa[1].hist(mouse_data['learning_time_in_trials'], bins=15, histtype='step', color='k')
    axa[1].set_xlabel('trials')
    axa[1].set_xlim((0, 2500))
    axa[1].set_xticks((0, 1200, 2400))
    axa[0].set_ylabel('number of mice')
    my.plot.despine(axa[0])
    my.plot.despine(axa[1])

    # Save
    f.savefig('HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS.svg')
    f.savefig('HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS.png', dpi=300)

    # Write stats
    with open('STATS__HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS', 'w') as fi:
        fi.write('skewness by trial: {:.4f}\n'.format(
            scipy.stats.skew(mouse_data['learning_time'].values)))
        fi.write('skewness by session: {:.4f}\n'.format(
            scipy.stats.skew(mouse_data['learning_time_in_trials'].values)))

    # Print back
    with open('STATS__HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS') as fi:
        print('STATS__HIST_LEARNING_TIME_BY_SESSIONS_AND_TRIALS\n')
        print(''.join(fi.readlines()))    




plt.show()