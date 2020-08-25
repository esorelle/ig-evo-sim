import os
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from datetime import datetime


###  HOW TO RUN  ###
# 0. dependencies -- python3 + python packages listed above
# 1. place this code in your working directory
# 2. open a terminal, navigate to the working directory
# 3. make sure simulation parameters below are set as desired (and save file)
# 4. at command prompt in terminal, type: python3 ig_evo_sim.py
# 5. simulation results + metadata will be output to timestamped directory


###  SET PARAMETERS FOR SIMULATION HERE  ###
# size of sampled cell population
n = 1000
# initial frequency of each B cell Ig isotype (IgM, IgA, IgG, IgE)
M, A, G, E = 0.48, 0.25, 0.25, 0.02
# isotype-specific probability of cell division in each discrete time interval
pM, pA, pG, pE = 0.85, 0.8585, 0.867, 0.867
# fixed cell death rate (uniform across isotypes)
death_thresh = 0.03
# number of rounds of cell division (discrete intervals) to simulate
rounds = 100
# number of simulation trials to run
trials = 10
# number of simulation trials for which to simulate clustering
cluster_sim_trials = 1
# interval (# of rounds) at which to simulate clustering
every_nth_round = 25


###  CREATES OUTPUT DIRECTORY  ###
dir = os.getcwd()
timestamp = datetime.now().strftime("%Y_%m_%d__%H%M%S")
save_path = os.path.join(dir, timestamp)
os.mkdir(save_path)
os.chdir(save_path)


###  SIMULATION CODE  ###
def clustering_sim_plot(cell_pop, round, trial=1):
    """Takes a cell pop list and plots simulated clustering
    of the cells f each isotype. Empirically observed clustering
    differences by Ig isotype are represented by arbitrary Ig-specific
    2D gaussian distributions."""

    fig, ax = plt.subplots()

    for cell in cell_pop:
        if cell == 'M':
            # IgM x and y gaussians (mu, sigma, n)
            xp = np.random.normal(-5, 2, 1)
            yp = np.random.normal(-2, 3, 1)
            ax.scatter(yp, xp, c='#1f77b4', s=3, label='IgM')
        elif cell == 'A':
            # IgA x and y gaussians
            xp = np.random.normal(3, 2, 1)
            yp = np.random.normal(2, 1.5, 1)
            ax.scatter(yp, xp, c='#ff7f0e', s=3, label='IgA')
        elif cell == 'G':
            # IgG x and y gaussians
            xp = np.random.normal(0, 1.5, 1)
            yp = np.random.normal(-5, 2.5, 1)
            ax.scatter(yp, xp, c='#2ca02c', s=3, label='IgG')
        elif cell == 'E':
            # IgE x and y gaussians
            xp = np.random.normal(4, 1.5, 1)
            yp = np.random.normal(-2, 1.5, 1)
            ax.scatter(yp, xp, c='#d62728', s=3, label='IgE')
        else:
            pass

    plt.xlabel('simulated_tSNE_1')
    plt.ylabel('simulated_tSNE_2')
    plt.legend(['IgM', 'IgA', 'IgG', 'IgE'])
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('#1f77b4')
    leg.legendHandles[1].set_color('#ff7f0e')
    leg.legendHandles[2].set_color('#2ca02c')
    leg.legendHandles[3].set_color('#d62728')
    plt.title('Simulated Clustering')
    plt.savefig('cluster_sim_trial_' + str(trial) + '_round_' + str(round) + '_' + timestamp + '.png', format='png', dpi=600)
    plt.close()


def create_cell_pop(n, M, A, G, E):
    """Initializes a B cell population of size n with IgM, IgA, IgG,
    and IgE cells. Default values for isotype proportions chosen based
    on empirical observation. IgM reflects B cells in various states
    before class-switch recombination. IgA, IgG, and IgE reflect
    proportions of corresponding class-switched memory and plasma cells."""

    cell_pop = []

    for cell in range(0, n+1):
        choice = rand.choices(['M', 'A', 'G', 'E'], weights=[M, A, G, E])
        cell_pop.append(choice[0])

    return cell_pop


def one_division_round(cell_pop, pM, pA, pG, pE, death_thresh):
    """Takes a cell population list as input, iterates through the list
    and applies isotype-specific probability of cell division. Default
    values for cell division probability are equal, but can be altered
    depending on model assumptions (for example, that class-switched
    cells may be more likely to divide than non class-switched cells).
    Assumes a constant cell death rate."""

    fixed_pop_size = len(cell_pop)
    new_cells = []

    for cell in cell_pop:
        death_chance = rand.random()

        if death_chance < death_thresh:
            cell_pop.remove(cell)
        else:
            prolif_prob = rand.random()
            if np.logical_and(cell=='M', pM > prolif_prob):
                new_cells.append('M')
            elif np.logical_and(cell=='A', pA > prolif_prob):
                new_cells.append('A')
            elif np.logical_and(cell=='G', pG > prolif_prob):
                new_cells.append('G')
            elif np.logical_and(cell=='E', pE > prolif_prob):
                new_cells.append('E')
            else:
                pass

    # create random sample of same size as original population
    after_prolif = cell_pop + new_cells

    # handle case where there is more cell death than proliferation in a round
    if len(after_prolif) < fixed_pop_size:
        fixed_pop_size = len(after_prolif)

    cell_pop = rand.choices(after_prolif, k=fixed_pop_size)

    # handle case where all cells are dead (avoid divide by len(cell_pop) = 0)
    if len(cell_pop) == 0:
        cell_pop = ['Dead']

    return cell_pop


def division_timecourse(cell_pop, rounds, trial=0):
    """Wraps one_division_round and records the frequency of each Ig isotype
    in the original population and after every round of cell division for
    r rounds. Returns isotype frequencies."""

    # create empty matrix with 4 arrays (M, A, G, E) for isotype frequency
    freq_mtx = np.empty((0,4), float)

    for round in range(0, rounds+1):
        pct_M = (float(cell_pop.count('M')) / len(cell_pop))
        pct_A = (float(cell_pop.count('A')) / len(cell_pop))
        pct_G = (float(cell_pop.count('G')) / len(cell_pop))
        pct_E = (float(cell_pop.count('E')) / len(cell_pop))

        freq_mtx = np.append(freq_mtx, np.array([[pct_M, pct_A, pct_G, pct_E]]), axis=0)

        # generates simulated clusters every nth round, for for first n trials ( < n)
        if trial < cluster_sim_trials:
            if np.logical_or(round == 0, round % every_nth_round == 0):
                clustering_sim_plot(cell_pop, round, trial)
            else:
                pass

        cell_pop = one_division_round(cell_pop, pM, pA, pG, pE, death_thresh)

    return freq_mtx


def plot_one_trial(freq_mtx):
    """Uses matplotlib to display evolution of Ig subtype frequencies
    over time for one run of the simulation."""

    rounds = range(0, np.shape(freq_mtx)[0])
    plt.plot(rounds, freq_mtx)
    plt.xlabel('Rounds of Cell Division')
    plt.ylabel('% of LCL Population')
    plt.legend(['IgM', 'IgA', 'IgG', 'IgE'])
    plt.title('Ig Isotype Composition over Time')
    plt.show()


def pop_sim(trials, rounds, start_pop=0):
    """Wraps division_timecourse and runs for a specified number of
    trials (instances). If no starting population is passed (start_pop=0),
    then each simulation run begins with a new call of create_cell_pop.
    If the output of create_cell_pop is passed as start_pop, then that
    population is used as the start-point for all simulation trials. Saves
    all trials for each isotype as a separate list of lists."""

    M_trials = []
    A_trials = []
    G_trials = []
    E_trials = []

    for trial in range(0, trials):

        # either create a new starting population for each trial
        if start_pop == 0:
            pop = create_cell_pop(n, M, A, G, E)
        # or use the same starting population passed by user
        else:
            pop = start_pop

        freq_mtx = division_timecourse(pop, rounds, trial)

        # store isotype frequencies in shape (trials, n_division_rounds)
        M_trials.append(freq_mtx[:,0])
        A_trials.append(freq_mtx[:,1])
        G_trials.append(freq_mtx[:,2])
        E_trials.append(freq_mtx[:,3])

    return M_trials, A_trials, G_trials, E_trials


def plot_all_trials(M_trials, A_trials, G_trials, E_trials):
    """Uses matplotlib to visualize all trials. Takes the isotype
    output lists from pop_sim and plots each trial for each isotype
    on the same axes. Not recommended for visualizing more than 5-10
    trials, as plot becomes overcrowded."""

    rounds = range(0, np.shape(M_trials)[1])

    for t in range(0, np.shape(M_trials)[0]):
        plt.plot(rounds, M_trials[t], linewidth=1)
        plt.plot(rounds, A_trials[t], linewidth=1)
        plt.plot(rounds, G_trials[t], linewidth=1)
        plt.plot(rounds, E_trials[t], linewidth=1)
        plt.gca().set_prop_cycle(None)

    plt.xlabel('Rounds of Cell Division')
    plt.ylabel('% of LCL Population')
    plt.legend(['IgM', 'IgA', 'IgG', 'IgE'])
    plt.title('Ig Isotype Composition over Time')
    plt.show()


def plot_trial_summary_stats(M_trials, A_trials, G_trials, E_trials):
    """An alternative to plot_all_trials that can be used for greater
    numbers of simulation trials. Takes the same input as plot_all_trials,
    but plots the mean and standard deviation (as shaded area) of all
    trial runs for each isotype."""

    rounds = range(0, np.shape(M_trials)[1])

    M_trials = np.array(M_trials)
    A_trials = np.array(A_trials)
    G_trials = np.array(G_trials)
    E_trials = np.array(E_trials)

    avg_M = np.mean(M_trials, axis=0)
    sd_M = np.std(M_trials, axis=0)
    avg_A = np.mean(A_trials, axis=0)
    sd_A = np.std(A_trials, axis=0)
    avg_G = np.mean(G_trials, axis=0)
    sd_G = np.std(G_trials, axis=0)
    avg_E = np.mean(E_trials, axis=0)
    sd_E = np.std(E_trials, axis=0)

    plt.plot(rounds, avg_M, linewidth=2)
    plt.fill_between(rounds, avg_M - sd_M, avg_M + sd_M, facecolor='#1f77b4', alpha=0.25)
    plt.plot(rounds, avg_A, linewidth=2)
    plt.fill_between(rounds, avg_A - sd_A, avg_A + sd_A, facecolor='#ff7f0e', alpha=0.25)
    plt.plot(rounds, avg_G, linewidth=2)
    plt.fill_between(rounds, avg_G - sd_G, avg_G + sd_G, facecolor='#2ca02c', alpha=0.25)
    plt.plot(rounds, avg_E, linewidth=2)
    plt.fill_between(rounds, avg_E - sd_E, avg_E + sd_E, facecolor='#d62728', alpha=0.25)

    plt.ylim(0, 1.05)
    plt.xlim(0, max(rounds))

    plt.title('Ig Isotype Composition over Time: {} Rounds'.format(max(rounds)))
    plt.xlabel('Rounds of Cell Division')
    plt.ylabel('% of LCL Population')
    plt.legend(['IgM', 'IgA', 'IgG', 'IgE'])
    plt.savefig('sim_plot_' + timestamp + '.png', format='png', dpi=600)
    plt.close()

    # plot probability distributions at last timepoint for all isotypes
    plt.hist(M_trials[:,-1], bins=33, weights=np.ones(len(M_trials[:,-1])) / len(M_trials[:,-1]), density=False, alpha=0.25)
    plt.hist(A_trials[:,-1], bins=33, weights=np.ones(len(A_trials[:,-1])) / len(A_trials[:,-1]), density=False, alpha=0.25)
    plt.hist(G_trials[:,-1], bins=33, weights=np.ones(len(G_trials[:,-1])) / len(G_trials[:,-1]), density=False, alpha=0.25)
    plt.hist(E_trials[:,-1], bins=33, weights=np.ones(len(E_trials[:,-1])) / len(E_trials[:,-1]), density=False, alpha=0.25)
    plt.title('Simulation Outcome Probability: {} Trials'.format(len(M_trials[:,-1])))
    plt.xlabel('Final Population Ig Composition')
    plt.xlim(0,1)
    plt.ylabel('Freqeuncy of Outcome')
    plt.legend(['IgM', 'IgA', 'IgG', 'IgE'])
    plt.savefig('outcome_hist_' + timestamp + '.png', format='png', dpi=600)
    plt.close()


###  FUNCTION CALL TO RUN SIMULATION  ###
pop = create_cell_pop(n, M, A, G, E)
M_trials, A_trials, G_trials, E_trials = pop_sim(trials, rounds, pop)
plot_trial_summary_stats(M_trials, A_trials, G_trials, E_trials)


###  WRITE SIMULATION PARAMETERS TO METADATA FILE  ###
metadata = open('_sim_run_metadata__' + timestamp + '.txt', "w+")
metadata.write('time of run: ' + datetime.now().strftime("%Y-%m-%d, %H:%M") + '\n')
metadata.write('rounds of cell division: ' + str(rounds) + '\n')
metadata.write('number of simulation trials: ' + str(trials) + '\n')
metadata.write('initial population size (# cells): ' + str(n) + '\n')

metadata.write('\n' + 'number of trials used for cluster simulation: ' + str(cluster_sim_trials) + '\n')
metadata.write('clusters simulated every ' + str(every_nth_round) + ' rounds of division' + '\n')

metadata.write('\n' + 'initial fraction IgM: ' + str(M) + '\n')
metadata.write('initial fraction IgA: ' + str(A) + '\n')
metadata.write('initial fraction IgG: ' + str(G) + '\n')
metadata.write('initial fraction IgE: ' + str(E) + '\n')

metadata.write('\n' + 'IgM proliferation probability: ' + str(pM) + '\n')
metadata.write('IgA proliferation probability: ' + str(pA) + '\n')
metadata.write('IgG proliferation probability: ' + str(pG) + '\n')
metadata.write('IgE proliferation probability: ' + str(pE) + '\n')

metadata.write('\n' + 'cell death probability: ' + str(death_thresh) + '\n')

metadata.close()


os.chdir('..')

###  END  ###
