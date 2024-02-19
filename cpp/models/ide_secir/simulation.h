/* 
* Copyright (C) 2020-2024 MEmilio
*
* Authors: Martin J Kuehn, Anna Wendler, Lena Ploetzke
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef IDE_SECIR_SIMULATION_H
#define IDE_SECIR_SIMULATION_H

#include "ide_secir/parameters.h"
#include "ide_secir/infection_state.h"
#include "ide_secir/model.h"
#include "memilio/config.h"
#include "memilio/utils/time_series.h"
#include <memory>
#include <cstdio>
#include <iostream>

namespace mio
{
namespace isecir
{

/**
 * run the simulation in discrete steps and report results.
 */
template <typename FP = double>
class Simulation
{

public:
    /**
     * @brief setup the Simulation for an IDE model.
     * @param[in] model An instance of the IDE model.
     * @param[in] t0 Start time.
     * @param[in] dt Step size of numerical solver.
     */
    Simulation(Model<FP> const& model, ScalarType t0 = 0., ScalarType dt = 0.1)
        : m_model(std::make_unique<Model<FP>>(model))
        , m_t0(t0)
        , m_dt(dt)
    {
    }

    /** 
     * Run the simulation from the current time to tmax.
     * @param tmax Time to stop.
     */
    void advance(ScalarType tmax)
    {
        mio::log_info("Simulating IDE-SECIR until t={} with dt = {}.", tmax, m_dt);
        m_model->initialize(m_dt);

        // for every time step:
        while (m_model->m_transitions.get_last_time() < tmax - m_dt / 2) {

            m_model->m_transitions.add_time_point(m_model->m_transitions.get_last_time() + m_dt);
            m_model->m_populations.add_time_point(m_model->m_populations.get_last_time() + m_dt);

            // compute_S:
            m_model->compute_susceptibles(m_dt);

            // compute flows:
            m_model->flows_current_timestep(m_dt);

            // compute D
            m_model->compute_deaths();

            // compute m_forceofinfection (only used for calculation of S and sigma_S^E in the next timestep!):
            m_model->update_forceofinfection(m_dt);

            // compute remaining compartments from flows
            m_model->other_compartments_current_timestep(m_dt);
            m_model->compute_recovered();
        }
    }

    /**
     * @brief Get the result of the simulation.
     * Return the number of persons in all #InfectionState%s.
     * @return The result of the simulation.
     */
    TimeSeries<double>& get_result()
    {
        return m_model->m_populations;
    }

    /**
     * @brief Get the result of the simulation.
     * Return the number of persons in all #InfectionState%s.
     * @return The result of the simulation.
     */
    const TimeSeries<double>& get_result() const
    {
        return m_model->m_populations;
    }

    /**
     * @brief Get the transitions between the different #InfectionState%s.
     * 
     * @return TimeSeries with stored transitions calculated in the simulation.
     */
    TimeSeries<ScalarType> const& get_transitions()
    {
        return m_model->m_transitions;
    }

    /**
     * @brief returns the simulation model used in simulation.
     */
    const Model<FP>& get_model() const
    {
        return *m_model;
    }

    /**
     * @brief returns the simulation model used in simulation.
     */
    Model<FP>& get_model()
    {
        return *m_model;
    }

    /**
     * @brief get the starting time of the simulation.
     * 
     */
    ScalarType get_t0()
    {
        return m_t0;
    }

    /**
     * @brief get the time step of the simulation.
     * 
     */
    ScalarType get_dt()
    {
        return m_dt;
    }

private:
    std::unique_ptr<Model<FP>> m_model; ///< Unique pointer to the Model simulated.
    ScalarType m_t0; ///< Start time used for simulation.
    ScalarType m_dt; ///< Time step used for numerical computations in simulation.
};

/**
 * @brief simulates a compartmental model
 * @param[in] t0 start time
 * @param[in] tmax end time
 * @param[in] dt initial step size of integration
 * @param[in] model an instance of a compartmental model
 * @return a TimeSeries to represent the final simulation result
 */
template <typename FP = ScalarType>
TimeSeries<ScalarType> simulate(FP t0, FP tmax, FP dt, Model<FP> const& m_model)
{
    m_model.check_constraints(dt);
    Simulation sim(m_model, t0, dt);
    sim.advance(tmax);
    return sim.get_result();
}

} // namespace isecir
} // namespace mio

#endif //IDE_SECIR_SIMULATION_H
