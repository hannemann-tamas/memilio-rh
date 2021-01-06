import epidemiology.secir as secir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def run_secir_simulation():
    # Define Comartment names
    compartments = ['Suscepted', 'Exposed', 'Carrying', 'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead', 'time']
    # Define age Groups
    groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
    # Define population of age groups
    populations = [30000, 60000, 150000, 200000, 120000, 100000]


    days = 100 # number of days to simulate
    t0 = 0
    dt = 0.1
    num_groups = len(groups)
    num_compartments = 8

    # Initialize Parameters
    params = secir.SecirParams(num_groups)

    # Set parameters
    for i in range(num_groups):
        # Compartment transition duration
        params.times[i].set_incubation(5.2)
        params.times[i].set_infectious_mild(6)
        params.times[i].set_serialinterval(4.2)
        params.times[i].set_hospitalized_to_home(12)
        params.times[i].set_home_to_hospitalized(5)
        params.times[i].set_hospitalized_to_icu(2)
        params.times[i].set_icu_to_home(8)
        params.times[i].set_icu_to_death(5)

        # Initial number of peaople in each compartment
        params.populations.set([i, secir.SecirCompartments.E], 100)
        params.populations.set([i, secir.SecirCompartments.C], 50)
        params.populations.set([i, secir.SecirCompartments.I], 50)
        params.populations.set([i, secir.SecirCompartments.H], 20)
        params.populations.set([i, secir.SecirCompartments.U], 10)
        params.populations.set([i, secir.SecirCompartments.R], 10)
        params.populations.set([i, secir.SecirCompartments.D], 0)
        params.populations.set_difference_from_group_total([i, secir.SecirCompartments.S],secir.SecirCategory.AgeGroup, i , populations[i])

        # Compartment transition propabilities
        params.probabilities[i].set_infection_from_contact(1.0)
        params.probabilities[i].set_carrier_infectability(0.67)
        params.probabilities[i].set_asymp_per_infectious(0.09)
        params.probabilities[i].set_risk_from_symptomatic(0.25)
        params.probabilities[i].set_hospitalized_per_infectious(0.2)
        params.probabilities[i].set_icu_per_hospitalized(0.25)
        params.probabilities[i].set_dead_per_icu(0.3)


    # set contact frequency matrix
    params.get_contact_patterns().cont_freq_mat[0].baseline = np.ones((num_groups,num_groups))*0.15
    params.get_contact_patterns().cont_freq_mat[0].minimum = np.ones((num_groups, num_groups)) * 0

    # Define Damping on Contacts
    params.get_contact_patterns().cont_freq_mat.add_damping(secir.Damping(np.ones((num_groups, num_groups))*0.9,30, 0, 0))

    # Apply mathematical constraints to parameters
    params.apply_constraints()

    # Run Simulation
    result = secir.simulate(t0, days, dt, params)
    print(result.get_last_value())

    num_time_points = result.get_num_time_points()
    result_array = result.as_ndarray()
    t = result_array[0, :]
    group_data = np.transpose(result_array[1:, :])

    #sum over all groups
    data = np.zeros((num_time_points,num_compartments))
    for i in range(num_groups):
        print(data[:,0])
        data += group_data[:, i*num_compartments:(i + 1) * num_compartments]

    # Plot Results
    datelist = np.array(pd.date_range(datetime(2020, 1, 27), periods=days, freq='D').strftime('%m-%d').tolist())
    datelist2 = np.array(pd.date_range(datetime(2020, 2, 3), periods=days, freq='D').strftime('%m-%d').tolist())

    tick_range = (np.arange(int(days / 10) +1 ) * 10)
    tick_range[-1] -=1
    fig, ax = plt.subplots()
    #ax.plot(t, data[:,0], label='#Suscepted')
    ax.plot(t, data[:,1], label='#Exposed')
    ax.plot(t, data[:,2], label='#Carrying')
    ax.plot(t, data[:,3], label='#Infected')
    ax.plot(t, data[:,4], label='#Hospitalzed')
    ax.plot(t, data[:,5], label='#In Intensive Care Units')
    #ax.plot(t, data[:,6], label='#Recovered')
    ax.plot(t, data[:,7], label='#Died')
    ax.set_title("SECIR model simulation")
    ax.set_xticks(tick_range)
    ax.set_xticklabels(datelist[tick_range],rotation=45)
    ax.legend()
    fig.tight_layout
    #fig.savefig('Secir_all' + name + '.pdf')

    ind_list = []
    for i in range(days):
        ind_list.append(np.argmin(np.abs(t-i)))

    new_sus = []
    for i in range(1,len(ind_list)):
        new_sus.append((data[ind_list[i-1],0] - data[ind_list[i], 0])/(t[ind_list[i]]-t[ind_list[i-1]]))

    fig, ax = plt.subplots()
    ax.plot(range(days-1), new_sus)
    ax.set_ylabel('Number of new cases')
    ax.set_title('Number of new Infections')
    ax.set_xticks(tick_range)
    ax.set_xticklabels(datelist[tick_range], rotation=45)
    fig.tight_layout
    #fig.savefig('new_infections' + name + '.pdf')


    if True:
        fig, ax = plt.subplots(4, 2, figsize=(12, 15))
        for i, title in zip(range(num_compartments), compartments):
            for j, group in zip(range(num_groups), groups):
                ax[int(np.floor(i / 2)),int(i % 2)].plot(t, group_data[:,j*num_compartments+i], label=group)
            ax[int(np.floor(i / 2)),int(i % 2)].set_title(title)
            ax[int(np.floor(i / 2)), int(i % 2)].legend()

            ax[int(np.floor(i / 2)), int(i % 2)].set_xticks(tick_range)
            ax[int(np.floor(i / 2)), int(i % 2)].set_xticklabels(datelist[tick_range], rotation=45)
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95)
        #fig.tight_layout()
        #fig.savefig('Secir_Groups' + name + '.pdf')


        fig, ax = plt.subplots(4, 2, figsize=(12, 15))
        for i, title in zip(range(num_compartments), compartments):
            ax[int(np.floor(i / 2)), int(i % 2)].plot(t, data[:, i])
            ax[int(np.floor(i / 2)), int(i % 2)].set_title(title)

            ax[int(np.floor(i / 2)), int(i % 2)].set_xticks(tick_range)
            ax[int(np.floor(i / 2)), int(i % 2)].set_xticklabels(datelist[tick_range], rotation=45)
        #ax[0,0].set_ylim(0, 10000)
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95)
        # fig.tight_layout()
        #fig.savefig('Secir_all_parts' + name + '.pdf')
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_secir_simulation()