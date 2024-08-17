## This directory generates behavior figures that don't require video
# Example plot of stimulus stream
 
import numpy as np
import matplotlib.pyplot as plt
import my.plot

target_rate_l = [5, 10]
target_temporal_std_l = list(10.0 ** np.array([-2, -1]))
#target_temporal_std_l = list(10.0 ** np.array([-3, -2]))

# This one is for eventplot
f2, axa2 = plt.subplots(len(target_rate_l), len(target_temporal_std_l), 
    sharex=True, sharey=True, figsize=(5, 1))
f2.subplots_adjust(hspace=0, wspace=.4, top=.95, left=.05, right=.95)

np.random.seed(6)

for target_rate in target_rate_l:
    for target_temporal_std in target_temporal_std_l:
        # Change of basis
        mean_interval = 1 / target_rate
        var_interval = target_temporal_std ** 2

        # Change of basis
        gamma_shape = (mean_interval ** 2) / var_interval
        gamma_scale = var_interval / mean_interval
        
        # Draw
        target_intervals = np.random.gamma(gamma_shape, gamma_scale, 100)
      
        # Convert intervals to times
        target_times = np.cumsum(target_intervals) - target_intervals[0]
        
        # Skipping some other processing here, such as dropping those
        # intervals below the sound duration, and flooring gaps at 1 sample

        # Choose a consistent duration
        target_times = target_times[target_times < 1.05]
        
        # Plot
        ax2 = axa2[
            target_rate_l.index(target_rate), 
            target_temporal_std_l.index(target_temporal_std),
            ]
        ax2.set_xlim((0, 1))
        ax2.set_xticks([])
        ax2.eventplot(target_times, color='k', clip_on=False)
        my.plot.despine(ax=ax2, which=('left', 'top', 'right'))
        ax2.set_yticks([])

my.plot.despine(axa2[0, 0], which=['bottom'])
my.plot.despine(axa2[0, 1], which=['bottom'])

f2.savefig("example_stimulus_streams.svg")
f2.savefig("example_stimulus_streams.png", dpi=300)
plt.show()