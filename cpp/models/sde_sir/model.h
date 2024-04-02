/* 
* Copyright (C) 2020-2024 MEmilio
*
* Authors: Nils Wassmuth, Rene Schmieding, Martin J. Kuehn
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

#ifndef MIO_SDE_SIR_MODEL_H
#define MIO_SDE_SIR_MODEL_H

#include "memilio/compartments/flow_model.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/utils/random_number_generator.h"
#include "sde_sir/infection_state.h"
#include "sde_sir/parameters.h"

namespace mio
{
namespace ssir
{

/********************
 * define the model *
 ********************/

using Flows = TypeList<Flow<InfectionState::Susceptible, InfectionState::Infected>,
                       Flow<InfectionState::Infected, InfectionState::Recovered>>;

class Model : public FlowModel<double, InfectionState, mio::Populations<double, InfectionState>, Parameters, Flows>
{
    using Base = FlowModel<double, InfectionState, mio::Populations<double, InfectionState>, Parameters, Flows>;

public:
    Model()
        : Base(Populations({InfectionState::Count}, 0.), ParameterSet())
    {
    }

    void get_flows(Eigen::Ref<const Eigen::VectorXd> pop, Eigen::Ref<const Eigen::VectorXd> y, double t,
                   Eigen::Ref<Eigen::VectorXd> flows) const
    {
        auto& params     = this->parameters;
        double coeffStoI = params.get<ContactPatterns>().get_matrix_at(t)(0, 0) *
                           params.get<TransmissionProbabilityOnContact>() / populations.get_total();

        double si = mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(rng, 0.0, 1.0);
        double ir = mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(rng, 0.0, 1.0);

        // Assuming that no person can change its InfectionState twice in a single time step,
        // take the minimum of the calculated flow and the source compartment, to ensure that
        // no compartment attains negative values.

        flows[get_flat_flow_index<InfectionState::Susceptible, InfectionState::Infected>()] = std::min(
            coeffStoI * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected] +
                sqrt(coeffStoI * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected]) /
                    sqrt(step_size) * si,
            y[(size_t)InfectionState::Susceptible] / step_size);

        flows[get_flat_flow_index<InfectionState::Infected, InfectionState::Recovered>()] = std::min(
            (1.0 / params.get<TimeInfected>()) * y[(size_t)InfectionState::Infected] +
                sqrt((1.0 / params.get<TimeInfected>()) * y[(size_t)InfectionState::Infected]) / sqrt(step_size) * ir,
            y[(size_t)InfectionState::Infected] / step_size);
    }

    ScalarType step_size = 0.1; ///< A step size of the model with which the stochastic process is realized.
    mutable RandomNumberGenerator rng;

private:
};

} // namespace ssir
} // namespace mio

#endif // MIO_SDE_SIR_MODEL_H
