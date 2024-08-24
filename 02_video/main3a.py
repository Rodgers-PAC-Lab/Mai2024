## This directory uses synced video to analyze body movement before/after HL
# main3a.py plots example frames and trajectories

import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import my
import my.plot
import datetime
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


## Load arena locations
arena_node_locs = pandas.read_pickle(
    os.path.join(repository_dir, 'synced_with_video', 'arena_node_locs'))


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



## Add columns to crossings
# Add columns to crossings (direction of ports, duration, outcome, and n_pokes)
# n_pokes is important because it defines the type of crossing
synced_big_crossings_df = extras.add_columns_to_crossings(
    pokes_with_crossings, synced_big_crossings_df, trial_data)


## Plots
PRETTY_PLOT_EXAMPLE_TRIALS = True

if PRETTY_PLOT_EXAMPLE_TRIALS:
    ## Pseudo-outer points
    # Octagon nodes
    outer = ['NNEO', 'ENEO', 'ESEO', 'SSEO', 'SSWO', 'WSWO', 'WNWO', 'NNWO']
    inner = ['NNEI', 'ENEI', 'ESEI', 'SSEI', 'SSWI', 'WSWI', 'WNWI', 'NNWI']
    pseudo = ['NNEP', 'ENEP', 'ESEP', 'SSEP', 'SSWP', 'WSWP', 'WNWP', 'NNWP']
    ports = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    # Calculate pseudo-outer nodes as a weighted average of outer and inner
    # TODO: do this by projecting onto the line connecting the nosepokes
    pseudo_l = []
    pseudo_keys_l = []
    for inner_name, outer_name in zip(inner, outer):
        outer_nodes = arena_node_locs.xs(outer_name, level=1)
        inner_nodes = arena_node_locs.xs(inner_name, level=1)
        pseudo_nodes = .65 * outer_nodes + .35 * inner_nodes
        pseudo_l.append(pseudo_nodes)
        pseudo_keys_l.append(inner_name[:-1] + 'P')
    
    # Concat
    pseudo_node_locs = pandas.concat(
        pseudo_l, keys=pseudo_keys_l, names=['node']).swaplevel().sort_index()
    arena_node_locs2 = pandas.concat(
        [arena_node_locs, pseudo_node_locs], verify_integrity=True).sort_index()
    
    
    ## Set example trials
    # Three row (mice): sham, bilateral, unilateral
    # Four cols (epochs): naive, expert, early HL, late HL
    example_sham = [
        ('Kidding_Octopus_085', datetime.date(2023, 6, 23), 16),
        ('Kidding_Octopus_085', datetime.date(2023, 7, 14), 47),
        ('Kidding_Octopus_085', datetime.date(2023, 8, 2), 54),
        ('Kidding_Octopus_085', datetime.date(2023, 8, 10), 42),
        ]        
    example_bilateral = [
        ('Party_Owl_096', datetime.date(2023, 6, 22), 38),
        ('Party_Owl_096', datetime.date(2023, 7, 19), 62),
        ('Party_Owl_096', datetime.date(2023, 8, 3), 14),
        ('Party_Owl_096', datetime.date(2023, 8, 10), 54),
        ]
    example_unilateral = [
        ('Guitar_089', datetime.date(2023, 6, 23), 16),
        ('Guitar_089', datetime.date(2023, 7, 18), 30),
        ('Guitar_089', datetime.date(2023, 8, 3), 4),
        ('Guitar_089', datetime.date(2023, 8, 10), 30),
        ]
        
    example_mice = [
        example_sham,
        example_bilateral,
        example_unilateral,
        ]
    pretty_group_l = ['sham mouse', 'bilateral mouse', 'unilateral mouse']
    pretty_epoch_l = ['early learning', 'late learning', 'early post-HL', 'late post-HL']
    
    ## Plot
    f, axa = plt.subplots(len(example_mice), 4, figsize=(8.5, 6))
    f.subplots_adjust(left=.05, right=.98, bottom=.02, top=.95, hspace=.15, wspace=.2)
    

    # Iterate over mice (rows)
    for n_row, mouse_trials in enumerate(example_mice):
        
        # Iterate over trials (cols)
        for n_col, (mouse_name, date_o, trial) in enumerate(mouse_trials):
            # Convert mouse_name and date_o to session_name
            session_row = session_data.set_index(
                'date', append=True).reset_index('session_name').loc[
                mouse_name].loc[date_o]
            session_name = session_row['session_name']
            n_session = session_row['n_session']
            n_from_manipulation = int(session_row['n_from_manipulation'])
            epoch = session_row.loc['epoch']
            this_video = session_row.loc['video_filename']
            hl_type = mouse_data.loc[mouse_name, 'manipulation']
            
            # Slice nodes
            this_node_locs = arena_node_locs2.loc[this_video]
            
            # Get ax
            ax = axa[n_row, n_col]
            if ax in axa[0]:
                ax.set_title(pretty_epoch_l[n_col])
            if ax in axa[:, 0]:
                ax.set_ylabel(pretty_group_l[n_row])

            # Slice pokes
            this_pokes = poke_data.xs(session_name, level=1).droplevel(
                'mouse').loc[trial].copy()
            prev_pokes = poke_data.xs(session_name, level=1).droplevel(
                'mouse').loc[trial-1].copy()

            # Stop plotting after rewarded poke
            rewarded_poke = this_pokes[
                this_pokes['reward_delivered'] == True].iloc[0]
            stop_frame = rewarded_poke['frame_number']
            
            # Start plotting after rewarded poke on previous trial
            prev_rewarded_poke = prev_pokes[
                prev_pokes['reward_delivered'] == True].iloc[0]
            start_frame = prev_rewarded_poke['frame_number']

            # Slice mouse loc from start_frame to stop_frame
            this_mouse_loc = synced_big_mouse_location_df.loc[
                session_name].droplevel('trial').loc[start_frame:stop_frame]

            # Slice crossings
            this_crossings = synced_big_crossings_df.loc[session_name].loc[trial]


            # Plot octagon
            for n in range(len(pseudo)):
                corner0 = pseudo[n]
                if n == len(pseudo) - 1:
                    corner1 = pseudo[0]
                else:
                    corner1 = pseudo[n + 1]

                # Connect the pseudos
                topl = this_node_locs.loc[[corner0, corner1], ['x', 'y']].values
                ax.plot(topl[:, 0], topl[:, 1], 'k-', lw=.75)
                
                # Connect pseudo to inner
                inner_corner = inner[n]
                topl = this_node_locs.loc[[corner0, inner_corner], ['x', 'y']].values
                ax.plot(topl[:, 0], topl[:, 1], 'k-', lw=.75)
                
                # Plot port
                port_loc = this_node_locs.loc[ports[n], ['x', 'y']].values
                ax.plot([port_loc[0]], [port_loc[1]], color='gray', marker='s', ms=4, alpha=.5)

            # Fix lims
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_frame_on(False)
            ax.axis('image')

            
            ## Plot trajectory
            # Downsample
            maxlen = 1000
            if len(this_mouse_loc) > maxlen:
                print('ds from {} to {}'.format(len(this_mouse_loc), maxlen))
                reidx = my.misc.take_equally_spaced(
                    range(len(this_mouse_loc)), maxlen)
                this_mouse_loc_topl = this_mouse_loc.iloc[reidx]
            else:
                this_mouse_loc_topl = this_mouse_loc
            
            # Plot as a gradienty
            ax.add_collection(
                my.plot.color_gradient(
                this_mouse_loc_topl['x'].values, 
                this_mouse_loc_topl['y'].values, 
                plt.cm.viridis_r,
                ))
            
            # Squares to mark start and stop
            ax.plot(this_mouse_loc['x'].iloc[0:1], this_mouse_loc['y'].iloc[0:1], marker='s', mew=1.5, mec='k', mfc='none')

            # Plot crossings
            center_out_crossings = this_crossings[this_crossings['typ'] == 'center-out'].copy()
            center_out_crossings = center_out_crossings.join(
                this_mouse_loc[['x', 'y']], on='first_frame_after')

            # Plot consummated crossings in green and unconsummated in gray
            consummated = center_out_crossings[center_out_crossings['n_pokes'] > 0]
            unconsummated = center_out_crossings[center_out_crossings['n_pokes'] == 0]
            ax.plot(
                unconsummated['x'], unconsummated['y'], 
                mec='r', mfc='none', marker='o', ls='none', ms=6, mew=1.5)

    # One more for a legend
    x_legend = ax.get_xlim()[0], ax.get_xlim()[1]
    y_legend = ax.get_ylim()[0] - 10, ax.get_ylim()[0] - 10
    ax.add_collection(
        my.plot.color_gradient(
        np.linspace(x_legend[0], x_legend[1], 50), 
        np.linspace(y_legend[0], y_legend[1], 50),
        plt.cm.viridis_r,
        ))
    
    # And a start square
    ax.plot(
        [x_legend[0]], [y_legend[0]],
        marker='s', mew=1.5, mec='k', mfc='none', clip_on=False)


    f.savefig('PRETTY_PLOT_EXAMPLE_TRIALS.svg')
    f.savefig('PRETTY_PLOT_EXAMPLE_TRIALS.png', dpi=300)


plt.show()
