## This directory generates behavior figures that don't require video
# main4a plots performance by sweep params, for naive, expert, and post-HL

import numpy as np
import scipy.stats
import os
import glob
import pandas
import datetime
import matplotlib.pyplot as plt
import my.plot


## Setup
# Figures
my.plot.manuscript_defaults()
my.plot.font_embed()

# Paths
repository_dir = os.path.expanduser(
    '~/mnt/cuttlefish/chris/data/20240817_ss_paper')
    
    
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


## Keep only 230201_SSS_sweep
session_data = session_data[
    session_data['protocol_name'] == '230201_SSS_sweep'].sort_index()

# Get corresponding sessions
keep_sessions = session_data.index.get_level_values('session_name')

# Drop from each
perf_metrics = perf_metrics.loc[
    perf_metrics['session_name'].isin(keep_sessions), :].sort_index()
trial_data = trial_data.loc[
    pandas.IndexSlice[:, keep_sessions], :].sort_index()
poke_data = poke_data.loc[
    pandas.IndexSlice[:, keep_sessions], :].sort_index()

# This totally removes some mice, so remove those from mouse_data
retained_mice = session_data.index.get_level_values('mouse').unique()
mouse_data = mouse_data.loc[retained_mice]


## Join epoch on trial_data
trial_data = trial_data.join(session_data['epoch'])
assert not trial_data['epoch'].isnull().any()


## Plot by sweep param
sweep_td = trial_data.copy()
sweep_td['is_hit'] = (sweep_td['outcome'] == 'correct').astype(int)

# Cut
swept_params = [
    'stim_target_rate',
    'stim_target_center_freq',
    'stim_target_amplitude', 
    'stim_target_temporal_std',
    ]

for swept_param in swept_params:
    sweep_td[swept_param + '_bin'] = pandas.cut(
        sweep_td[swept_param], bins=12, right=False, labels=False)
    
    # check
    vc = sweep_td[swept_param + '_bin'].value_counts()
    assert vc.max() < 1.1 * vc.min()


## Plots
PPT_SWEEP_PERF = True


## Plot PPT
if PPT_SWEEP_PERF:
    stats_l = []
    stats_keys_l = []
    
    for epoch in ['naive', 'expert', 'hl_bilateral']:
        # Select trials
        these_td = sweep_td[sweep_td['epoch'] == epoch]
        
        f, axa = plt.subplots(1, len(swept_params), figsize=(8.5, 3), sharey=True)
        f.subplots_adjust(wspace=.6, left=.09, right=.96, bottom=.25, top=.98)
        #~ f.suptitle(epoch)
        for swept_param in swept_params:
            # group
            agged = these_td.groupby(
                ['mouse', swept_param + '_bin']
                )['rcp'].mean().unstack('mouse')        
            
            # Use this to get the x-axis labels
            xticklabels = sweep_td.groupby(swept_param + '_bin')[swept_param].median()
            
            # Get ax
            ax = axa[
                swept_params.index(swept_param)]
            
            # Set title
            if False: #ax in axa[0]:
                ax.set_title(swept_param)

            # Get topl
            topl = agged
            
            # Convert RCP to PPT
            topl = topl + 1
            assert not topl.isnull().any().any()
            
            # Run some stats
            var_name = topl.index.name
            for_stats = topl.stack().rename('perf').reset_index()
            
            # Run as a linreg
            linreg_res = scipy.stats.linregress(for_stats[var_name], for_stats['perf'])
            pval = linreg_res.pvalue
            sigstr = my.stats.pvalue_to_significance_string(pval)
            
            # Run as ANOVA
            aov_res = my.stats.anova(
                for_stats, 'perf ~ {} + mouse'.format(var_name))
            aov_pval = aov_res['pvals']['p_' + var_name]
            aov_sigstr = my.stats.pvalue_to_significance_string(aov_pval)
            
            # Plot
            ax.plot(xticklabels, topl.values, color='k', lw=.75, alpha=.5)
            ax.plot(xticklabels, topl.mean(axis=1).values, 'k', lw=3)
            #~ ax.legend(topl.columns)
            
            # Pretty
            my.plot.despine(ax)
            
            ax.set_xticks((xticklabels.min(), xticklabels.max()))
            ax.set_xlim((xticklabels.min(), xticklabels.max()))
            
            if swept_param == 'stim_target_rate':
                ax.set_xticklabels((
                    int(np.round(xticklabels.min(), 1)), 
                    int(np.round(xticklabels.max(), 1)),
                    ))            
                ax.set_xlabel('repetition rate (Hz)')
            
            elif swept_param == 'stim_target_center_freq':
                ax.set_xticklabels((
                    int(np.round(.001 * xticklabels.min(), 1)), 
                    int(np.round(.001 * xticklabels.max(), 1)),
                    ))
                ax.set_xlabel('center frequency (kHz)')
            
            if swept_param == 'stim_target_amplitude':
                # According to my notes on 2023-03-07, an amplitude of -2 is
                # 70 dB SPL assuming 3 kHz bandwidth
                ax.set_xticklabels((
                    int(np.round(110 + 20 * xticklabels.min(), 1)), 
                    int(np.round(110 + 20 * xticklabels.max(), 1)),
                    ))
                ax.set_xlabel('sound level (dB SPL)')
                
            elif swept_param == 'stim_target_temporal_std':
                ax.set_xticklabels((
                    int(np.round(1000 * 10 ** xticklabels.min(), 1)), 
                    int(np.round(1000 * 10 ** xticklabels.max(), 1)),
                    ))
                ax.set_xlabel('irregularity (ms)')

            if '*' in aov_sigstr:
                ax.text(np.mean(ax.get_xlim()), 1.3, aov_sigstr, ha='center', va='center')
            else:
                ax.text(np.mean(ax.get_xlim()), 1.5, aov_sigstr, ha='center', va='center', fontsize=12)
            ax.plot(ax.get_xlim(), [4, 4], 'k--', lw=1)
            
            # Store stats
            # Multiply slope by 11 to get the full range
            stats_l.append((pval, aov_pval, topl.shape[1], linreg_res.slope * 11))
            stats_keys_l.append((epoch, swept_param))
        
        # Pretty
        ax.set_ylim((1, 4.5))
        ax.set_yticks((1, 2, 3, 4))
        axa[0].set_ylabel('ports poked per trial\n(lower is better)')
        
        
        f.savefig('PPT_SWEEP_PERF_{}.svg'.format(epoch))
        f.savefig('PPT_SWEEP_PERF_{}.png'.format(epoch), dpi=300)


    ## Concat stats
    stats_df = pandas.DataFrame.from_records(
        stats_l, columns=['p', 'aov_p', 'n', 'slope'], 
        index=pandas.MultiIndex.from_tuples(
        stats_keys_l, names=['epoch', 'swept_param']))
    stats_df['sig'] = stats_df['p'] < 0.05
    stats_df['aov_sig'] = stats_df['aov_p'] < 0.05

    ## Write stats
    with open('STATS__PPT_SWEEP_PERF', 'w') as fi:
        fi.write('2-way ANOVA (mouse + parameter) on each group vs swept_param\n')
        fi.write(str(stats_df))
    
    # Write back
    with open('STATS__PPT_SWEEP_PERF') as fi:
        print(''.join(fi.readlines()))

plt.show()