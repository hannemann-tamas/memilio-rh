/* 
* Copyright (C) 2020-2024 MEmilio
*
* Authors: Daniel Abele, Jan Kleinert, Martin J. Kuehn
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

#ifndef ODESIR_MODEL_H
#define ODESIR_MODEL_H

#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/epidemiology/age_group.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/epidemiology/contact_matrix.h"
#include "ode_sir/infection_state.h"
#include "ode_sir/parameters.h"

namespace mio
{
namespace osir
{

/********************
    * define the model *
    ********************/
template <typename FP = ScalarType>
class Model : public mio::CompartmentalModel<FP, InfectionState, mio::Populations<FP, InfectionState>, Parameters<FP>>
{
    using Base = mio::CompartmentalModel<FP, InfectionState, mio::Populations<FP, InfectionState>, Parameters<FP>>;

public:
    Model()
        : Base(mio::Populations<FP, InfectionState>({InfectionState::Count}, 0.), typename Base::ParameterSet())
    {
    }

    void get_derivatives(Eigen::Ref<const Vector<FP>> pop, Eigen::Ref<const Vector<FP>> y, FP t,
                         Eigen::Ref<Vector<FP>> dydt) const override
    {
        auto& params     = this->parameters;
        double coeffStoI = params.template get<ContactPatterns<FP>>().get_matrix_at(t)(0, 0) *
                           params.template get<TransmissionProbabilityOnContact<FP>>() / this->populations.get_total();

        dydt[(size_t)InfectionState::Susceptible] =
            -coeffStoI * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected];
        dydt[(size_t)InfectionState::Infected] =
            coeffStoI * y[(size_t)InfectionState::Susceptible] * pop[(size_t)InfectionState::Infected] -
            (1.0 / params.template get<TimeInfected<FP>>()) * y[(size_t)InfectionState::Infected];
        dydt[(size_t)InfectionState::Recovered] =
            (1.0 / params.template get<TimeInfected<FP>>()) * y[(size_t)InfectionState::Infected];
    }
};

} // namespace osir
} // namespace mio

#endif // ODESIR_MODEL_H
