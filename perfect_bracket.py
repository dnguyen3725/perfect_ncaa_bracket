from IPython.core.debugger import Tracer
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bigfloat import *

pick_stats = '2016_espn_picks.csv'

prec = 1000

use_real_stats = True

# Parse pick statistics
pick_prob = []
pick_prob.append({})
pick_prob.append({})
pick_prob.append({})
pick_prob.append({})
pick_prob.append({})
pick_prob.append({})
with open(pick_stats, 'rb') as f:
    f_iter = csv.reader(f)
    for row in f_iter:
                    
        # Determine number of rounds in this row
        n_rounds = len(row)/2
        
        # Loop through each round and add data to pick probabilities
        for i in range(0,n_rounds):
        
            team = row[i*2]
            prob = row[i*2+1]
            
            if len(team) > 0:
                if use_real_stats:
                    pick_prob[i][team] = BigFloat(prob, context=precision(prec))
                else:
                    pick_prob[i][team] = BigFloat(pow(0.5,i+1), context=precision(prec))
       
# Compute probability of guessing each round perfectly
p_perfect_round = []
p_perfect_round_corr = []
p_perfect_tourn = BigFloat(1.0, context=precision(prec))
for i in range(0, 6):
    
    p_perfect_round.append(BigFloat(1.0, context=precision(prec)))
    p_perfect_round_corr.append(BigFloat(1.0, context=precision(prec)))
    
    for team in pick_prob[i]:
    
        p_perfect_round[i] = mul(p_perfect_round[i], pick_prob[i][team], context=precision(prec))
        
        if i == 0:
            pick_prob_corr = pick_prob[i][team]
        else: 
            pick_prob_corr = div(pick_prob[i][team], pick_prob[i-1][team], context=precision(prec))
        p_perfect_round_corr[i] = mul(p_perfect_round_corr[i], pick_prob_corr, context=precision(prec))
        
    print 'Round {}: {}% (1 in {})'.format(i+1, mul(p_perfect_round[i],100), div(1,p_perfect_round[i]))
    
    p_perfect_tourn = mul(p_perfect_tourn,p_perfect_round_corr[i],context=precision(prec))
   
print 'Tournament: {}% (1 in {})'.format(mul(p_perfect_tourn,100), div(1,p_perfect_tourn))

# Convert round probabilities to floats for plotting
p_perfect_round_float = np.zeros(len(p_perfect_round))
p_perfect_round_corr_float = np.zeros(len(p_perfect_round_corr))
for i in range(0, len(p_perfect_round)):
    p_perfect_round_float[i] = float(p_perfect_round[i])
    p_perfect_round_corr_float[i] = float(p_perfect_round_corr[i])

sns.set_context("poster")

# Plot barchart of perfect pick probabilities per round
plt.figure(1)
sns.set_style("darkgrid")
x = ["R64","R32","Sweet 16","Elite 8","Final 4","Championship"]
fig = sns.barplot(x, p_perfect_round_float*100, palette="Blues_d")
fig.set_ylabel('Uncorrelated Probability of Perfect Picks (%)')
for p in fig.patches:
    height = p.get_height()
    fig.text(p.get_x(), height*1.1, '{:.3}%'.format(height))
fig.set_yscale('log')

# Plot barchart of correlated perfect pick probabilities per round
plt.figure(2)
fig = sns.barplot(x, p_perfect_round_corr_float*100, palette="Blues_d")
fig.set_ylabel('Correlated Probability of Perfect Picks (%)')
fig.set_yscale('log')
for p in fig.patches:
    height = p.get_height()
    fig.text(p.get_x(), height*1.1, '{:.3}%'.format(height))


# Generate a precision 1
prec_1 = BigFloat(1.0, context=precision(prec))
    
# Compute number of trials to have a 50/50 shot of getting a perfect bracket
p_desired = 0.5
p_fail = sub(prec_1, BigFloat(p_desired, context=precision(prec)), context=precision(prec))
p_inv = sub(prec_1, p_perfect_tourn, context=precision(prec))
p_log_fail = log(p_fail, context=precision(prec))
p_log_inv = log(p_inv, context=precision(prec))
trials_req = div(p_log_fail, p_log_inv, context=precision(prec))
print 'Trials required for {}% chance: {}'.format(p_desired*100, float(trials_req))

# Compute geometric probability of getting a perfect tournament
trials = np.linspace(1.0, 1e17, num=1000)
p_geom = np.zeros(len(trials))
for i in range(0, len(trials)):
    p_fail = sub(prec_1, p_perfect_tourn, context=precision(prec))
    p_fail_trials = pow(p_fail, BigFloat(trials[i], context=precision(prec)), context=precision(prec))
    p_succ = sub(prec_1, p_fail_trials , context=precision(prec))
    p_geom[i] = float(p_succ)
plt.figure(3)
fig = plt.plot(trials/1e12, p_geom*100)
plt.xlabel('Number of Brackets (Trillions)')
plt.ylabel('Probability of a Perfect Bracket (%)')

# Show figures
plt.show()