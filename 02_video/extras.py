import numpy as np
import pandas
import my.plot
import scipy.stats

def generate_box_port_dir_df():
    """Returns the direction of each port.
    
    Returns: DataFrame
        index: integers
        columns: box, port, dir
            box: name of parent pi
            port: name of port
            dir: direction of port, with 0 indicating north and 90 east
    """
    box2port_name2port_dir = {
        'rpi_parent01': {
            'rpi09_L': 315,
            'rpi09_R': 0,
            'rpi10_L': 45,
            'rpi10_R': 90,
            'rpi11_L': 135,
            'rpi11_R': 180,
            'rpi12_L': 225,
            'rpi12_R': 270,
        },
        'rpi_parent02': {
            'rpi07_L': 315,
            'rpi07_R': 0,
            'rpi08_L': 45,
            'rpi08_R': 90,
            'rpi05_L': 135,
            'rpi05_R': 180,
            'rpi06_L': 225,
            'rpi06_R': 270,
        },    
        'rpiparent03': {
            'rpi01_L': 225,
            'rpi01_R': 270,
            'rpi02_L': 315,
            'rpi02_R': 0,
            'rpi03_L': 45,
            'rpi03_R': 90,
            'rpi04_L': 135,
            'rpi04_R': 180,
        }, 
        'rpiparent04': {
            'rpi18_L': 90,
            'rpi18_R': 135,
            'rpi19_L': 180,
            'rpi19_R': 225,
            'rpi20_L': 270,
            'rpi20_R': 315,
            'rpi21_L': 0,
            'rpi21_R': 45,
        },
    }    

    # Parse
    ser_d = {}
    for box_name, port_name2port_dir in box2port_name2port_dir.items():
        ser = pandas.Series(port_name2port_dir, name='dir')
        ser.index.name = 'port'
        ser_d[box_name] = ser

    # Concat
    box_port_dir_df = pandas.concat(ser_d, names=['box']).reset_index()
    
    # Convert degrees to cardinal directions
    box_port_dir_df['cardinal'] = box_port_dir_df['dir'].replace({
        0: 'N',
        45: 'NE',
        90: 'E',
        135: 'SE',
        180: 'S',
        225: 'SW',
        270: 'W',
        315: 'NW',
        })
    
    return box_port_dir_df

def add_columns_to_crossings(pokes_with_crossings, synced_big_crossings_df, trial_data):
    """Adds columns to synced_big_crossings_df
    
    Adds the following: 
        n_pokes
        rewarded_port, previously_rewarded_port
        rewarded_region, previously_rewarded_region
        dist_from_rp, dist_from_prp
        adjacent_to_rp, adjacent_to_prp
        duration
        outcome
    
    Returns: synced_big_crossings_df with columns added
    """
    # Count pokes on each crossing
    n_pokes_per_crossing = pokes_with_crossings.groupby(
        ['session_name', 'trial', 'crossing']).size().rename('n_pokes')

    # Join onto crossings
    synced_big_crossings_df = synced_big_crossings_df.join(n_pokes_per_crossing)

    # Fillna with 0
    # TODO: check that these are actually zero-poke crossings, and not just
    # crossings where we don't have poke data (is this possible?)
    # TODO: find out why there are sometimes pokes on a center-in
    synced_big_crossings_df['n_pokes'] = (
        synced_big_crossings_df['n_pokes'].fillna(0).astype(int))

    # Join rp and the prp on the crossings
    synced_big_crossings_df = synced_big_crossings_df.join(
        trial_data['rewarded_port'].droplevel('mouse'))
    synced_big_crossings_df = synced_big_crossings_df.join(
        trial_data['previously_rewarded_port'].droplevel('mouse'))

    # Translate previousy_rewarded_port into a region number
    # This has to match the numbering when regions were assigned
    chamber_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    box_port_dir_df = generate_box_port_dir_df()
    box_port_dir_df['region'] = box_port_dir_df['cardinal'].map(
        pandas.Series(dict([(v, k) for k, v in enumerate(chamber_order)])))

    # Join prev_region
    port2region = box_port_dir_df.set_index('port')['region']
    synced_big_crossings_df['previously_rewarded_region'] = (
        synced_big_crossings_df['previously_rewarded_port'].map(port2region))
    synced_big_crossings_df['rewarded_region'] = (
        synced_big_crossings_df['rewarded_port'].map(port2region))

    # As a check, can see whether the first region_before is the previously_rewarded_region
    # This won't work well if the ITI was long or the mouse was fast
    # xxx = synced_big_crossings_df.groupby(['session_name', 'trial']).first()[['region_before', 'previously_rewarded_region']]
    # (xxx['region_before'] == xxx['previously_rewarded_region']).mean()

    # Calculate dist_from_prp (for cycling)
    synced_big_crossings_df['dist_from_prp'] = (
        synced_big_crossings_df['region_after'] - 
        synced_big_crossings_df['previously_rewarded_region'])
    synced_big_crossings_df['dist_from_rp'] = (
        synced_big_crossings_df['region_after'] - 
        synced_big_crossings_df['rewarded_region'])

    # Normalize to (-4, 3)
    synced_big_crossings_df['dist_from_prp'] = np.mod(
        synced_big_crossings_df['dist_from_prp'] + 4, 8) - 4
    synced_big_crossings_df['dist_from_rp'] = np.mod(
        synced_big_crossings_df['dist_from_rp'] + 4, 8) - 4

    # Mark adjacent
    synced_big_crossings_df['adjacent_to_prp'] = synced_big_crossings_df['dist_from_prp'].isin([-1, 1])
    synced_big_crossings_df['adjacent_to_rp'] = synced_big_crossings_df['dist_from_rp'].isin([-1, 1])

    # Calculate duration within each region
    synced_big_crossings_df['duration'] = (
        synced_big_crossings_df['end_of_crossing'] - 
        synced_big_crossings_df['first_frame_after'])

    # Join on outcome
    synced_big_crossings_df = synced_big_crossings_df.join(
        trial_data['outcome'].droplevel('mouse'))
    
    return synced_big_crossings_df

def add_crossing_number_to_kinematics(
    synced_big_crossings_df, synced_big_kinematics_df):
    """Add crossing number to synced_big_kinematics_df
    
    We do this so we can assess paths during crossings later
    And potentially also so we can blank out PRP time
    TODO: do this back when we first defined crossings using kinematics
    """

    # Take the first_frame_after and put that on the index with crossing as a value
    first_frame_after = synced_big_crossings_df['first_frame_after'].rename(
        'frame').reset_index().set_index(
        ['session_name', 'frame'])['crossing'].rename('crossing_ffill')

    # Do the same for last_frame_before
    # Have to drop 'trial' here or it won't join correctly, because a crossing
    # can span a trial boundary
    last_frame_before = synced_big_crossings_df['end_of_crossing'].rename(
        'frame').reset_index().set_index(
        ['session_name', 'frame'])['crossing'].rename('crossing_bfill')

    # Join onto kinematics
    synced_big_kinematics_df = synced_big_kinematics_df.join(
        pandas.concat(
        [first_frame_after, last_frame_before], axis=1, verify_integrity=True))

    # Ensure they all got joined
    assert len(first_frame_after) == len(synced_big_kinematics_df['crossing_ffill'].dropna())

    # Fillna the crossing number for frames between crossings
    crossing_by_frame_l = []
    crossing_by_frame_keys_l = []
    for session_name in synced_big_kinematics_df.index.levels[0]:
        # ffill from first_frame_after and bfill from end_of_crossing
        sess = synced_big_kinematics_df.loc[session_name].copy()
        sess['ffilled'] = sess['crossing_ffill'].ffill()
        sess['bfilled'] = sess['crossing_bfill'].bfill()
        
        # Error check that these measure never differ when both are non-null
        bad_mask = (
            (sess['bfilled'] != sess['ffilled']) & 
            ~sess['bfilled'].isnull() & 
            ~sess['ffilled'].isnull()
        )
        assert not np.any(bad_mask)
        
        # Keep only where they are both non-null (so they should agree)
        crossing_by_frame = sess['ffilled'][
            ~sess['bfilled'].isnull() & ~sess['ffilled'].isnull()]
        
        # Store (to avoid settingwithcopywarning)
        crossing_by_frame_l.append(crossing_by_frame.rename('crossing'))
        crossing_by_frame_keys_l.append(session_name)

    # Join the crossing by_frame onto big_kinematics
    # All frames labeled with this crossing will have region equal to the 
    # "region_after" column. The labeled frames *follow* the crossing itself
    # So if you want crossings through the center, select center-in
    # This will be null where a crossing cannot be assigned, which generally happens
    # at the beginning and end of each session, I think
    to_join = pandas.concat(
        crossing_by_frame_l, keys=crossing_by_frame_keys_l, names=['session_name'])
    synced_big_kinematics_df = synced_big_kinematics_df.join(to_join)

    # Drop the now unnecessary columns
    synced_big_kinematics_df = synced_big_kinematics_df.drop(
        ['crossing_ffill', 'crossing_bfill'], axis=1)
    
    return synced_big_kinematics_df

def quantify_paths_during_crossings(
    synced_big_crossings_df, 
    synced_big_kinematics_df, 
    session_df):
    """Quantify distance traveled and straightness for each center crossing
    
    """
    # Convert px to mm
    synced_big_kinematics_df['x_mm'] = (
        synced_big_kinematics_df['x'] * session_df['mm_per_px'].droplevel('mouse'))
    synced_big_kinematics_df['y_mm'] = (
        synced_big_kinematics_df['y'] * session_df['mm_per_px'].droplevel('mouse'))

    # Calculate distance head traveled on each frame
    dist_by_frame_l = []
    dist_by_frame_keys_l = []
    for session_name, session_bkdf in synced_big_kinematics_df.groupby('session_name'):
        # Weirdly NaN on the first frame is converted to zero
        dist_by_frame = np.sqrt((session_bkdf[['x_mm', 'y_mm']].diff() ** 2).sum(axis=1))
        dist_by_frame_l.append(dist_by_frame)

    # Store
    synced_big_kinematics_df['dist_travelled'] = pandas.concat(dist_by_frame_l)

    # Quantify tortuosity
    # tortuosity in general is hard to calculate
    # arc-to-chord ratio is straightforward, but a circle is infinite
    # mean(delta(body_velocity_angle)) doesn't work well during periods of
    # slow motion .. would need to resample into period of equal distance travelled
    gobj = synced_big_kinematics_df.groupby(['session_name', 'crossing'])
    dist_travelled = gobj['dist_travelled'].sum()
    start_pos = gobj[['x_mm', 'y_mm']].first()
    end_pos = gobj[['x_mm', 'y_mm']].last()
    chord_len = np.sqrt(((end_pos - start_pos) ** 2).sum(axis=1)).rename('chord_len')
    tortuosity = dist_travelled / chord_len

    # Ceiling it because it will be infinite for a perfect circle
    # Tortuosity has an ugly distribution very heavy tailed
    # 1/tortuosity nicely uniform in [0, 1]
    # crossings of duration 1 frame will have infinite tortuosity
    tortuosity[tortuosity > 300] = 300

    # Store
    # Some crossings have null dist_travelled, chord_len, and inv_tortu
    # I think because they got dropped from kinematics for some reason (maybe
    # only at the end or beginning of the session?)
    # center-out crossings (those whose path is entirely in the side chamber)
    # have high tortuosity (are circular)
    # center-in crossings (those entirely within the center) have low tortuosity
    # these tend to have inv_tortu at zero (I guess the circular ones) or near 0.6
    synced_big_crossings_df = synced_big_crossings_df.join(
        pandas.concat([
        dist_travelled, chord_len, 1 / tortuosity.rename('straightness')], axis=1),
        on=['session_name', 'crossing'])
    
    return synced_big_crossings_df

def extract_center_out_crossings(synced_big_crossings_df):
    """Extract center-out crossings and add metadata
    
    Starts with row in synced_big_crossings_df for which 'typ' == 'center-out'
    Calculates the following:
        dist_from_prev : angular distance from previous center-out crossing
            Will be null for first crossing in each session
        is_cycling : True when dist_from_prev in [-1, 1]
        into_prp : True when region_after == previously_rewarded_region
        is_dup : True when this is the second (or later) crossing into that
            region on this trial
        is_first : True for the first center-out of each trial that is not
            into the prp
        has_poke : True if n_pokes > 0 on that crossing
    
    Returns: center_outs
        A subset of the rows from synced_big_crossings_df, with the
        columns above added.
    """
    # Get center outs only
    center_outs = synced_big_crossings_df[
        synced_big_crossings_df['typ'] == 'center-out'].copy()

    # Calculate dist_from_prev (for cycling)
    # This is DIFFERENT from dist_from_prp calculates in extras, because this
    # is only valid for center-out crossings, and it is relative to the previous
    # center-out crossing
    # This will be null for the first center-out in each session
    center_outs['dist_from_prev'] = (
        center_outs['region_after'] - 
        center_outs['region_after'].groupby('session_name').shift()
        )

    # Normalize to (-4, 3)
    center_outs['dist_from_prev'] = np.mod(
        center_outs['dist_from_prev'] + 4, 8) - 4

    # Mark cycling entries as those with dist_from_prev in [-1, 1]
    center_outs['is_cycling'] = center_outs['dist_from_prev'].isin([-1, 1])

    # Mark the ones into prp
    center_outs['into_prp'] = (
        center_outs['region_after'] == center_outs['previously_rewarded_region'])

    # Drop duplicates - keep only the first center-out into each region per trial
    center_outs_no_repeats = center_outs.reset_index('crossing').groupby(
        ['session_name', 'trial', 'region_after']
        ).first().reset_index('region_after').set_index('crossing', append=True)

    # Use this to assign into center_outs
    center_outs['is_dup'] = True
    center_outs.loc[center_outs_no_repeats.index, 'is_dup'] = False

    # Mark the first non-prp of each trial
    firsts = center_outs.loc[~center_outs['into_prp']].reset_index(
        'crossing').groupby(
        ['session_name', 'trial']).first().set_index(
        'crossing', append=True)
    assert not firsts['is_dup'].any()
    center_outs['is_first'] = False
    center_outs.loc[firsts.index, 'is_first'] = True

    # Mark those with and without poke
    center_outs['has_poke'] = center_outs['n_pokes'] > 0

    # Return
    return center_outs

def categorize_center_out_crossings(center_outs):
    """Categorize types of center-out crossing
    
    Arguments
        center_outs : Indexed by session_name, trial, crossing
    
    Counts the following types of center-out crossing per trial:
        n_center_outs : All center-out crossings
        
        n_cycling : All center-out crossings that are adjacent to the previously
            entered port (i.e., cycling)
        
        is_dup : All center-out crossings into a port that was already visited
            at least once before on this trial.
        
        into_prp : Ell center-out crossings into the previously rewarded port
            (including duplicates)
        
        consumms : Excluding duplicates and entries into the PRP,
            all entries with a poke
        
        ducks : Excluding duplicates and entries into the PRP,
            all entries without a poke
        
        ducks + consumms : Sum of ducks and consumms
    
    Returns : crossing_counts_by_trial
        Indexed by session_name, trial
    """
    # Keep results here
    crossing_counts_by_trial_d = {}

    # Count total crossings
    crossing_counts_by_trial_d['n_center_outs'] = center_outs.groupby(
        ['session_name', 'trial']).size().rename('n_center_outs')

    # Count the total number of cycling entries per trial, excluding dups
    # For this, we don't care about excluding PRP or duplicates, purely
    # the probability that the entry is adjacent to the previous one
    crossing_counts_by_trial_d['n_cycling'] = center_outs.loc[
        center_outs['is_cycling']
        ].groupby(['session_name', 'trial']).size()

    # Entries into a previously visited chamber (excluded from all that follows)
    # We exclude these later because it messes up the chance rate
    crossing_counts_by_trial_d['is_dup'] = center_outs.loc[
        center_outs['is_dup']
        ].groupby(['session_name', 'trial']).size()

    # Entries into PRP (excluded from all that follows)
    # We exclude these later because it messes up the chance rate
    crossing_counts_by_trial_d['into_prp'] = center_outs.loc[
        center_outs['into_prp']
        ].groupby(['session_name', 'trial']).size()

    # Entries into PRP with poke
    # This is used to confirm what this looks like in a behavior-only plot
    crossing_counts_by_trial_d['into_prp_consumm'] = center_outs.loc[
        center_outs['into_prp'] &
        center_outs['has_poke']
        ].groupby(['session_name', 'trial']).size()

    # Entries with poke
    crossing_counts_by_trial_d['consumms'] = center_outs.loc[
        ~center_outs['into_prp'] & 
        ~center_outs['is_dup'] & 
        center_outs['has_poke']
        ].groupby(['session_name', 'trial']).size()

    # Entries without poke
    crossing_counts_by_trial_d['ducks'] = center_outs.loc[
        ~center_outs['into_prp'] & 
        ~center_outs['is_dup'] & 
        ~center_outs['has_poke']
        ].groupby(['session_name', 'trial']).size()

    
    ## Concat these metrics
    crossing_counts_by_trial = pandas.DataFrame.from_dict(
        crossing_counts_by_trial_d).fillna(0).astype(int)

    # Also sum all non-dup
    crossing_counts_by_trial['ducks + consumms'] = (
        crossing_counts_by_trial['ducks'] + crossing_counts_by_trial['consumms'])

    return crossing_counts_by_trial

def compute_time_in_each_region(synced_big_kinematics_df):
    """Compute time in chambers, straightness, etc
    
    Computes the following for each trial
        time_in_center, time_in_sides, time_total
            Uses the frame count within each region on each trial
    """
    # Count frames in each region
    total_time_by_region = synced_big_kinematics_df.groupby(
        ['session_name', 'trial', 'region']).size().unstack(
        'region').fillna(0).astype(int)

    # Convert to seconds
    total_time_by_region = total_time_by_region / 30

    # Quantify time in center and each side
    quantified_time_by_trial = pandas.DataFrame.from_dict({
        'time_in_center': total_time_by_region.loc[:, 8],
        'time_in_sides': total_time_by_region.drop(8, axis=1).sum(axis=1),
        })
    quantified_time_by_trial['time_total'] = total_time_by_region.sum(axis=1)


    return quantified_time_by_trial

def run_stats(data_wrt_start=None, data_wrt_hl=None):
    """Runs stats wrt start and HL
    
    data_wrt_start : DataFrame or None
        index: n_session. Should include only sessions to analyze, eg 0-15
        columns: MultiIndex (manipulation2, mouse)
        values: Non-interpolated (okay to include nan)

    data_wrt_hl : same, except indexed wrt hearing loss, eg -5-11
    
    During learning:
    * This analysis uses data_wrt_start
    * perf ~ mouse + n_session
        * learning_pval
        * learning_start: mean over first three sessions provided
        * learning_stop: mean over last three sessions provided
    
    After HL:
    * This analysis uses data_wrt_hl.loc[1:], so only data after HL
    * perf ~ mouse + manipulation2 + n_from_manipulation
        p-values
        * hl_recovery_group_pval (pval on manipulation2)
        * hl_recovery_time_pval (pval on n_from_manipulation)
        
        Effect sizes
        * hl_recovery_before: mean over three days before HL
        * hl_recovery_soonafter: mean over three days after HL (1:3)
        * hl_recovery_longafter: mean over three last days provided
    
    Slope after recovery, each group:
    * This analysis uses data_wrt_hl.loc[1:], separately by group
    * perf ~ mouse + n_from_manipulation, within each group
        p-values
        * hl_recovery_group_slope_pval_{groupname} (pval on n_from_manipulation)
        
        Effect sizes as in previous ANOVA:
        * hl_recovery_group_before_{groupname}
        * hl_recovery_group_soonafter_{groupname}
        * hl_recovery_group_longafter_{groupname}
    
    Peri-HL difference, each group
        * This analysis uses data_wrt_hl, separately by group
        * Compares the three days before to all provided days after (1:), 
            using paired t-test
        * peri_HL__{groupname}__pval
    
    Significant difference during recovery, each pair of groups
    * This analysis uses data_wrt_hl.loc[1:], each pair of groups
    * perf ~ mouse + manipulation2 + n_from_manipulation, each pair of groups
        p-values
        * hl_recovery__inter_group__{group1}_VS_{group2}__pval
        
        effect size (sign relative to sorted order of group names)
        * hl_recovery__inter_group__{group1}_VS_{group2}__fit
    """
    ## Stats
    stats = {}
    
    
    ## Analyses wrt start
    if data_wrt_start is not None:
        
        ## ANOVA during learning
        # Groups have not been experimentally differentiated yet so do not
        # include manipulation (although there are likely diffs between cohorts)
        learning_data = data_wrt_start.stack(
            future_stack=True).stack(future_stack=True).rename(
            'perf').reset_index()
        learning_aov_res = my.stats.anova(
            learning_data, 'perf ~ mouse + n_session')
        
        # learning effect is the p-value on n_session during learning
        stats['learning_pval'] = learning_aov_res['pvals']['p_n_session']
        
        # Also grab the effect on first three days and last three days
        stats['learning_start'] = data_wrt_start.iloc[:3].mean().mean()
        stats['learning_stop'] = data_wrt_start.iloc[-3:].mean().mean()
    
    
    ## Analyses wrt HL
    if data_wrt_hl is not None:
        
        ## ANOVA after HL 
        # Note we include only data after HL, not during baseline
        hl_recovery_data = data_wrt_hl.loc[1:].stack(
            future_stack=True).stack(future_stack=True).rename(
            'perf').reset_index()
        hl_recovery_res = my.stats.anova(
            hl_recovery_data.dropna(), 
            'perf ~ mouse + manipulation2 + n_from_manipulation')
        
        # learning effect is the p-value on n_session during learning
        stats['hl_recovery_group_pval'] = (
            hl_recovery_res['pvals']['p_manipulation2'])
        stats['hl_recovery_time_pval'] = (
            hl_recovery_res['pvals']['p_n_from_manipulation'])
        
        # Also grab the effect on first three days and last three days
        stats['hl_recovery_before'] = data_wrt_hl.loc[-3:-1].mean().mean()
        stats['hl_recovery_soonafter'] = data_wrt_hl.loc[1:3].mean().mean()
        stats['hl_recovery_longafter'] = data_wrt_hl.iloc[-3:].mean().mean()
        
        
        ## Check for significant slope after recovery in each group
        for group_name in data_wrt_hl.columns.levels[0]:
            # Only after HL
            this_group = data_wrt_hl.loc[1:, group_name]
            this_group_data = this_group.stack(
                future_stack=True).rename('perf').reset_index()
            
            # AOV
            this_group_aov_res = my.stats.anova(
                this_group_data, 'perf ~ mouse + n_from_manipulation')
            
            # Extract p-vals
            stats['hl_recovery_group_slope_pval_{}'.format(group_name)] = (
                this_group_aov_res['pvals']['p_n_from_manipulation'])
            
            # Extract effect size
            stats['hl_recovery_group_before_{}'.format(group_name)] = (
                data_wrt_hl.loc[-3:-1, group_name].mean().mean())
            stats['hl_recovery_group_soonafter_{}'.format(group_name)] = (
                data_wrt_hl.loc[1:3].loc[:, group_name].mean().mean())
            stats['hl_recovery_group_longafter_{}'.format(group_name)] = (
                data_wrt_hl.iloc[-3:].loc[:, group_name].mean().mean())


        ## Check for an pre/post effect in each group
        for group_name1 in data_wrt_hl.columns.levels[0]:
            # Grab data from this group
            this_data = data_wrt_hl.loc[:, group_name1]
            
            # Mean data in pre and post epochs
            # Alternatively, could keep days as replicates
            pre_data = this_data.loc[-3:-1].mean()
            post_data = this_data.loc[1:].mean()
            
            # Concat for AOV
            pre_vs_post_data = pandas.concat(
                [pre_data, post_data], 
                keys=['pre', 'post'], 
                names=['epoch']).rename('perf').unstack('epoch')
            
            # Actually a paired t-test works for this
            ttest_res = scipy.stats.ttest_rel(
                pre_vs_post_data['pre'], 
                pre_vs_post_data['post'], 
                )
            
            # Store
            stats['peri_HL__{}__pval'.format(group_name1)
                ] = ttest_res.pvalue
        
        
        ## Check for significant differences between groups during recovery
        # This analysis can give counter-intuitive results, but I'm not sure
        # how better to post-hoc compare between groups in a repeated-measures
        for group_name1 in data_wrt_hl.columns.levels[0]:
            for group_name2 in data_wrt_hl.columns.levels[0]:
                # Skip if we've already done this comparison
                if group_name1 >= group_name2:
                    continue
                
                # Grab data from the two groups
                this_data = data_wrt_hl.loc[:, [group_name1, group_name2]]
                longitudinal_data = this_data.loc[1:]
                
                # Run the longitudinal anova
                aov_data = longitudinal_data.stack(
                    future_stack=True).stack(future_stack=True).rename(
                    'perf').reset_index()
                this_group_aov_res = my.stats.anova(
                    aov_data, 
                    'perf ~ mouse + manipulation2 + n_from_manipulation')                
                
                # Store the p-value
                key = 'hl_recovery__inter_group__{}_VS_{}__pval'.format(
                    group_name2, group_name1)
                stats[key] = this_group_aov_res['pvals']['p_manipulation2']
                
                # Store the fit
                key = 'hl_recovery__inter_group__{}_VS_{}__fit'.format(
                    group_name2, group_name1)
                
                # The fit is keyed by group_name2 and it is wrt group_name1
                # (with group_name2 > group_name1 lexicographically)
                key2 = 'fit_manipulation2[T.{}]'.format(group_name2)
                stats[key] = this_group_aov_res['fit'][key2]

    return stats

def make_plot(ax, this_topl, this_topl_err, metric, group2color):
    """Helper function to make plot wrt start and plot wrt HL
    
    Plots the mean and shaded error bars for this_topl and this_topl_err
    
    Uses `metric` to set ylim and yticks, and optionally plot a chance line
    """
    # Always plot these (sometimes there is a 'none' group)
    groups_to_plot = [
        'hl_bilateral',
        'hl_sham',
        'hl_unilateral',
        ]
    
    # Plot
    for group_name in groups_to_plot:
        color = group2color[group_name]
        
        ax.plot(this_topl[group_name], color=color)
        ax.fill_between(
            x=this_topl[group_name].index,
            y1=this_topl[group_name] - this_topl_err[group_name],
            y2=this_topl[group_name] + this_topl_err[group_name],
            color=color, alpha=.5, lw=0
            )

    
    ## Plot chance line
    if metric in ['consumms', 'ducks + consumms']:
        ax.plot(this_topl[group_name].index[[0, -1]], [4, 4], 'k--', lw=1)
    
    elif metric == 'p_cycling':
        # Technically chance rate is 2/8, but this sort of assumes that the
        # mouse might re-enter the chamber it just entered, which is of course
        # pretty rare, so 2/7 might be more fair. 
        ax.plot(this_topl[group_name].index[[0, -1]], [0.25, 0.25], 'k--', lw=1)

    elif metric == 'frac_returned':
        # Chance is 1/8
        ax.plot(this_topl[group_name].index[[0, -1]], [0.125, 0.125], 'k--', lw=1)
        
    
    ## Set ylim and yticks
    if metric == 'ducks':
        ax.set_ylim((0, 2))
        ax.set_yticks((0, 1, 2))
    
    elif metric == 'into_prp':
        ax.set_ylim((0, 2))
        ax.set_yticks((0, 1, 2))
    
    elif metric == 'frac_returned':
        ax.set_ylim((0, 1))
        ax.set_yticks((0, 0.5, 1))
    
    elif metric == 'is_dup':
        ax.set_ylim((0, 8))
        ax.set_yticks((0, 4, 8))
    
    elif metric == 'consumms':
        ax.set_ylim((0, 6))
        ax.set_yticks((0, 3, 6))
    
    elif metric == 'ducks + consumms':
        ax.set_ylim((0, 6))
        ax.set_yticks((0, 3, 6))
    
    elif metric == 'p_cycling':
        ax.set_ylim((0, 1))
        ax.set_yticks((0, 0.5, 1))

    # Pretty
    ax.set_xlim(this_topl[group_name].index[[0, -1]])
    my.plot.despine(ax)

def write_out_stats(stats_df):
    with open('STAT__QUANTIFY_ENTRIES_THROUGHOUT_LEARNING_AND_HL', 'w') as fi:
        # Duplicates decrease over learning
        fi.write(
            'change in duplicates over learning: '
            '\n  start={:.3f}\tstop={:.3f}\n  p_time={:.2g}\n'.format(
            stats_df.loc['learning_start', 'is_dup'],
            stats_df.loc['learning_stop', 'is_dup'],
            stats_df.loc['learning_pval', 'is_dup'],
            ))
        
        # After HL, duplicates do not change over time or between groups
        fi.write(
            'change in duplicates after HL: '
            '\n  before={:.3f}\tearly={:.3f}\tlate={:.3f} \n  p_group={:.2g}\tp_time={:.2g}\n'.format(
            stats_df.loc['hl_recovery_before', 'is_dup'],
            stats_df.loc['hl_recovery_soonafter', 'is_dup'],
            stats_df.loc['hl_recovery_longafter', 'is_dup'],
            stats_df.loc['hl_recovery_group_pval', 'is_dup'],
            stats_df.loc['hl_recovery_time_pval', 'is_dup'],
            ))    
        
        # Write out each metric
        for metric in stats_df.columns:
            # Heading
            fi.write('\nFOR THE METRIC {}\n---\n'.format(metric))
            
            # Values before HL, and whether it changes over time
            fi.write(
                'before HL (grand means over time and p-value for effect of time): '
                '\n\tstart={:.3f}\tstop={:.3f}\n\tp_time={:.2g}\n'.format(
                stats_df.loc['learning_start', metric],
                stats_df.loc['learning_stop', metric],
                stats_df.loc['learning_pval', metric],
                ))
            
            # After HL, totals are different between groups
            fi.write(
                'after HL (grand means over time and p-value for effect of time and group): '
                '\n\tbefore={:.3f}\tearly={:.3f}\tlate={:.3f} \n\tp_group={:.2g}\tp_time={:.2g}\n'.format(
                stats_df.loc['hl_recovery_before', metric],
                stats_df.loc['hl_recovery_soonafter', metric],
                stats_df.loc['hl_recovery_longafter', metric],
                stats_df.loc['hl_recovery_group_pval', metric],
                stats_df.loc['hl_recovery_time_pval', metric],
                ))
            
            # Pre-vs-post HL, whether it changes for each group
            fi.write(
                'pre-vs-post HL (t-test for each group): '
                )
            fi.write(
                '\n\tp_sham={:.2g}\tp_uni={:.2g}\tp_bi={:.2g}\n'.format(
                stats_df.loc['peri_HL__hl_sham__pval', metric],
                stats_df.loc['peri_HL__hl_unilateral__pval', metric],
                stats_df.loc['peri_HL__hl_bilateral__pval', metric],
                ))
            
            # After HL, totals for each group
            fi.write(
                'after HL (individual group means):\n')
            fi.write(
                '\tSHM\tbefore={:.3f}\tearly={:.3f}\tlate={:.3f}\tp_slope={:.2g}\n'.format(
                stats_df.loc['hl_recovery_group_before_hl_sham', metric],
                stats_df.loc['hl_recovery_group_soonafter_hl_sham', metric],
                stats_df.loc['hl_recovery_group_longafter_hl_sham', metric],
                stats_df.loc['hl_recovery_group_slope_pval_hl_sham', metric],
                ))       
            fi.write(
                '\tUNI\tbefore={:.3f}\tearly={:.3f}\tlate={:.3f}\tp_slope={:.2g}\n'.format(
                stats_df.loc['hl_recovery_group_before_hl_unilateral', metric],
                stats_df.loc['hl_recovery_group_soonafter_hl_unilateral', metric],
                stats_df.loc['hl_recovery_group_longafter_hl_unilateral', metric],
                stats_df.loc['hl_recovery_group_slope_pval_hl_unilateral', metric],
                ))    
            fi.write(
                '\tBI \tbefore={:.3f}\tearly={:.3f}\tlate={:.3f}\tp_slope={:.2g}\n'.format(
                stats_df.loc['hl_recovery_group_before_hl_bilateral', metric],
                stats_df.loc['hl_recovery_group_soonafter_hl_bilateral', metric],
                stats_df.loc['hl_recovery_group_longafter_hl_bilateral', metric],
                stats_df.loc['hl_recovery_group_slope_pval_hl_bilateral', metric],
                ))    

            # After HL, post-hoc diffs between totals for each group
            fi.write(
                'post-hoc group diffs after HL:\n')
            fi.write((
                '\tSHM vs BI :\t\tfit={:.3f}\tp={:.2g}\n'
                '\tUNI vs BI :\t\tfit={:.3f}\tp={:.2g}\n'
                '\tUNI vs SHM:\t\tfit={:.3f}\tp={:.2g}\n').format(
                stats_df.loc['hl_recovery__inter_group__hl_sham_VS_hl_bilateral__fit', metric],
                stats_df.loc['hl_recovery__inter_group__hl_sham_VS_hl_bilateral__pval', metric],
                stats_df.loc['hl_recovery__inter_group__hl_unilateral_VS_hl_bilateral__fit', metric],
                stats_df.loc['hl_recovery__inter_group__hl_unilateral_VS_hl_bilateral__pval', metric],
                stats_df.loc['hl_recovery__inter_group__hl_unilateral_VS_hl_sham__fit', metric],
                stats_df.loc['hl_recovery__inter_group__hl_unilateral_VS_hl_sham__pval', metric],
                ))             

