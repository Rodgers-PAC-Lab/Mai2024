## This directory generates behavior figures that don't require video
# main3 plots performance aligned to hearing loss

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
    
    
## Load data from main1
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


## Include only hearing loss mice
mouse_data = mouse_data.loc[~mouse_data['hearing loss date'].isnull()]
retained_mice = mouse_data.index
perf_metrics = perf_metrics.loc[retained_mice]
session_data = session_data.loc[retained_mice]
trial_data = trial_data.loc[retained_mice]
poke_data = poke_data.loc[retained_mice]


## Check that protocol was consistent peri HL 
n_protocols_per_mouse = session_data[
    session_data['n_from_manipulation'] >= -10].groupby(
    ['mouse', 'protocol_name']).size().groupby('mouse').size()
assert (n_protocols_per_mouse == 1).all()


## Replace
mouse_data['hearing loss type'] = mouse_data['hearing loss type'].replace({
    'uni': 'unilateral',
    'bi': 'bilateral',
    'left': 'unilateral',
    'right': 'unilateral',
    })


## Recalculate perf metrics, now including non-consummatory PRP
# TODO: move to another scripts
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
lbpd_ranked = lbpd_unstacked.rank(
    method='first', axis=1).stack().astype(int) - 1

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


## Reindex perf_metrics by n_from_manipulation
perf_metrics = perf_metrics.reset_index().set_index(
    ['mouse', 'n_from_manipulation'])


## Plots
PLOT_RCP_BY_HL_TYPE_AGGREGATED_MICE = True


## like PLOT_RCP_BY_HL_TYPE but error bars instead of individual mice
if PLOT_RCP_BY_HL_TYPE_AGGREGATED_MICE:
    ## Plot vs hearing loss type
    
    # Plot colors
    group2color = {
        'unilateral': 'blue',
        'bilateral': 'red',
        'sham': 'black',
        }

    
    ## Repeat this plot once including PRP and once without
    for plot_type in ['normal', 'wprp']:
        
        ## Group by hearing loss type
        groups = mouse_data.groupby('hearing loss type').groups
        
        # Turn gobj.groups into a DataFrame
        series_l = []
        series_keys_l = []
        for group_label, group_idx in groups.items():
            series_l.append(pandas.Series(group_idx, name='mouse'))
            series_keys_l.append(group_label)
        grouping_idx = pandas.concat(
            series_l, keys=series_keys_l, names=['group', 'idx'])
        grouping_idx = grouping_idx.reset_index().drop('idx', axis=1).set_index(
            'mouse')['group']
        
        
        ## Join those groups onto perf_metrics
        # Join
        this_pm = my.misc.join_level_onto_index(perf_metrics, grouping_idx)
        
        # Drop any null groups
        this_pm = this_pm.drop(np.nan, level='group')
        
        # Slice out data of interest
        # This is the only part that differs with plot_type
        if plot_type == 'normal':
            this_pm = this_pm['rcp'].unstack().loc[:, -10:10]
        else:
            this_pm = this_pm['rcp_wprp'].unstack().loc[:, -10:10]
        
        # convert to ppt
        this_pm = this_pm + 1
        
        
        ## Stats
        stats = {}
        
        # ANOVA after HL 
        # Note we include only data after HL, not during baseline
        hl_recovery_data = this_pm.loc[:, 0:10].stack().rename(
            'perf').reset_index()
        hl_recovery_res = my.stats.anova(
            hl_recovery_data, 'perf ~ mouse + group + n_from_manipulation')
        
        # learning effect is the p-value on n_session during learning
        stats['hl_recovery_group_pval'] = hl_recovery_res['pvals']['p_group']
        stats['hl_recovery_time_pval'] = hl_recovery_res['pvals']['p_n_from_manipulation']
        
        # Check for significant slope after recovery in each group
        for group_name in this_pm.index.levels[0]:
            # Only after HL
            this_group = this_pm.loc[group_name].loc[:, 1:10]
            this_group_data = this_group.stack().rename('perf').reset_index()
            
            # AOV
            this_group_aov_res = my.stats.anova(
                this_group_data, 'perf ~ mouse + n_from_manipulation')
            
            # Extract p-vals
            stats['hl_recovery_group_slope_pval_{}'.format(group_name)] = this_group_aov_res[
                'pvals']['p_n_from_manipulation']
            

        # Check for an pre/post effect in each group
        for group_name1 in this_pm.index.levels[0]:
            # Grab data from the two groups
            this_group = this_pm.loc[group_name1]
            
            # Mean data in pre and post epochs
            # Alternatively, could keep days as replicates
            pre_data = this_group.loc[:, -3:-1].mean(axis=1)
            post_data = this_group.loc[:, 1:10].mean(axis=1)
            
            # Concat for AOV
            pre_vs_post_data = pandas.concat(
                [pre_data, post_data], 
                keys=['pre', 'post'], names=['epoch']).rename('perf').unstack('epoch')
            
            # Actually a paired t-test works for this
            ttest_res = scipy.stats.ttest_rel(
                pre_vs_post_data['pre'].values, 
                pre_vs_post_data['post'].values, 
                )
            
            # Store
            stats['peri_HL__{}__pval'.format(group_name1)
                ] = ttest_res.pvalue
        
        
        # Check for significant differences between groups during recovery
        for group_name1 in this_pm.index.levels[0]:
            for group_name2 in this_pm.index.levels[0]:
                # Skip if we've already done this comparison
                if group_name1 >= group_name2:
                    continue
                
                # Grab data from the two groups
                longitudinal_data = this_pm.loc[[group_name1, group_name2]].loc[:, 1:10]
                
                # Run the longitudinal anova
                aov_data = longitudinal_data.stack().rename(
                    'perf').reset_index()
                this_group_aov_res = my.stats.anova(
                    aov_data, 'perf ~ mouse + group + n_from_manipulation')                


                # Store the p-value
                key = 'hl_recovery__inter_group__{}_VS_{}__pval'.format(
                    group_name2, group_name1)
                stats[key] = this_group_aov_res['pvals']['p_group']
                
                # Store the fit
                key = 'hl_recovery__inter_group__{}_VS_{}__fit'.format(
                    group_name2, group_name1)
                
                # The fit is keyed by group_name2 and it is wrt group_name1
                # (with group_name2 > group_name1 lexicographically)
                key2 = 'fit_group[T.{}]'.format(group_name2)
                stats[key] = this_group_aov_res['fit'][key2]


        ## Plot
        f, ax = plt.subplots(figsize=(4.3, 2.1))
        f.subplots_adjust(left=.16, top=.94, bottom=.25, right=.65)

        # Plot each group
        for group_label in this_pm.index.levels[0]:
            # Get color
            color = group2color[group_label]
            
            # Extract what to plot
            topl = this_pm.loc[group_label].T.copy()

            # Plot mean
            ax.plot(topl.index.values + 0.5, topl.mean(1).values, color=color)#, lw=2)
            
            # Plot SEM
            ax.fill_between(
                x=topl.index.values + 0.5, 
                y1=topl.mean(1).values - topl.sem(1).values, 
                y2=topl.mean(1).values + topl.sem(1).values, 
                color=color, lw=0, alpha=.5)

        # Chance line
        if plot_type == 'normal':
            ax.plot((-10, 10), [4, 4], 'k--', lw=1)
        
        elif plot_type == 'wprp':
            # Chance rate is now higher
            ax.plot((-10, 10), [4.5, 4.5], 'k--', lw=1, clip_on=False)

        # Pretty
        ax.set_ylim((1, 4.5))
        ax.set_xlim((-10, 10))
        ax.set_xticks((-10, 0, 10))
        ax.set_yticks((1, 2, 3, 4))
        my.plot.despine(ax)
        ax.set_ylabel('ports poked per trial\n(lower is better)')
        ax.set_xlabel('sessions from hearing loss')

        # Legend
        f.text(
            .67, .62, 'n = {} sham'.format(len(groups['sham'])), 
            color='k', ha='left')
        f.text(
            .67, .71, 'n = {} unilateral'.format(len(groups['unilateral'])), 
            color='b', ha='left')
        f.text(
            .67, .8, 'n = {} bilateral'.format(len(groups['bilateral'])), 
            color='r', ha='left')

        # Save figs
        f.savefig('PLOT_RCP_BY_HL_TYPE_AGGREGATED_MICE_{}.svg'.format(plot_type))
        f.savefig('PLOT_RCP_BY_HL_TYPE_AGGREGATED_MICE_{}.png'.format(plot_type), dpi=300)
        
        
        ## Dump stats
        stats_filename = (
            'STATS__PLOT_RCP_BY_HL_TYPE_AGGREGATED_MICE_{}'.format(plot_type))
        
        with open(stats_filename, 'w') as fi:
            n_mice = this_pm.groupby('group').size()
            fi.write('n = {} total\n{}\n\n'.format(n_mice.sum(), n_mice))
            fi.write(
                'p-value on before vs after:\n'
                '  SHM={:.4f}; BI={:.4f}; UNI={:.4f}\n'.format(
                stats['peri_HL__sham__pval'],
                stats['peri_HL__bilateral__pval'],
                stats['peri_HL__unilateral__pval'],
                ))
            fi.write(
                'aov during recovery p-value on\n'
                '  group={:.4f} and time={:.4f}\n'.format(
                stats['hl_recovery_group_pval'],
                stats['hl_recovery_time_pval'],
                ))
            fi.write(
                'slope during recovery p-value on\n'
                '  SHM={:.4f}; BI={:.4f}; UNI={:.4f}\n'.format(
                stats['hl_recovery_group_slope_pval_sham'],
                stats['hl_recovery_group_slope_pval_bilateral'],
                stats['hl_recovery_group_slope_pval_unilateral'],
                ))
            fi.write(
                'between groups during recovery\n'
                '  p-value SHMvBI={:.4f}, UNIvBI={:.4f}, UNIvSHM={:.4f}\n'.format(
                stats['hl_recovery__inter_group__sham_VS_bilateral__pval'],
                stats['hl_recovery__inter_group__unilateral_VS_bilateral__pval'],
                stats['hl_recovery__inter_group__unilateral_VS_sham__pval'],
                )) 
        
        # Write back
        with open(stats_filename) as fi:
            print(''.join(fi.readlines()))
    

plt.show()
