import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# The output of each factory
fac1 = np.array((0,0,9,5,3,2,1,0,0,0,0,0))
fac2 = np.array((0,0,0,0,0,3,2,1,1,0,0,0))
fac3 = np.array((0,5,5,6,6,7,4,2,1,0.5,0,0))


# a matrix to store the all, for ease later on.
allfac = np.hstack((fac1, fac2, fac3))

num_days = 1000 # we collect data for 1000 days

data = np.zeros((12, num_days)) # preallocate matrix to store our data


fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(allfac)
plt.xlabel('Time (hours)')
plt.ylabel('Factory output')
ax.
plt.legend('Factory 1', 'Factory 2', 'Factory 3')
plt.title('Individual Factories')

'''
for d == 1:num_days
which_factories = boolean(randi([0 1], 1, 3)); # Randomly decide which factory discharges today
output_for_day = sum(allfactories(:, which_factories), 2); # sum the output of active factories
data(:, d) = output_for_day;
end

plt.subplot(2, 1, 2);
plt.plot(output_for_day);
hold on;
plt.plot(allfactories(:, which_factories), '--', 'Color', [0.5 0.5 0.5]);
hold off
plt.xlabel('Time (hours)');
plt.ylabel('Amount in lake');
plt.legend('Total Output', 'Individual Output');
plt.title('Data from day 1000')


'''