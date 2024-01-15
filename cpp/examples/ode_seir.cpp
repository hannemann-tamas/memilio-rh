/* 
* Copyright (C) 2020-2024 MEmilio
*
* Authors: Daniel Abele, Martin J. Kuehn
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
#include "ode_seir/model.h" // IWYU pragma: keep
#include "ode_seir/infection_state.h" // IWYU pragma: keep
#include "ode_seir/parameters.h"
#include "memilio/compartments/simulation.h" // IWYU pragma: keep
#include "memilio/utils/logging.h"
#include "memilio/math/adapt_rk.h"
#include "memilio/utils/time_series.h" // IWYU pragma: keep
#include "memilio/utils/time_series_to_file.h"
#include <fstream>



int main()
{
    mio::set_log_level(mio::LogLevel::debug);

    double t0   = 0;
    double tmax = 100;
    double dt   = 0.01;

    mio::log_info("Simulating SEIR; t={} ... {} with dt = {}.", t0, tmax, dt);

    mio::oseir::Model<double> model;

    double total_population                                                                            = 10000;
    model.populations[mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)]   = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}]  = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}] = 100;
    model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Susceptible)}] =
        total_population -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Exposed)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Infected)}] -
        model.populations[{mio::Index<mio::oseir::InfectionState>(mio::oseir::InfectionState::Recovered)}];
    // suscetible now set with every other update
    // params.nb_sus_t0   = params.nb_total_t0 - params.nb_exp_t0 - params.nb_inf_t0 - params.nb_rec_t0;
    model.parameters.set<mio::oseir::TimeExposed<double>>(5.2);
    model.parameters.set<mio::oseir::TimeInfected<double>>(6);
    model.parameters.set<mio::oseir::TransmissionProbabilityOnContact<double>>(0.04);
    model.parameters.get<mio::oseir::ContactPatterns>().get_baseline()(0, 0) = 10;

    model.check_constraints();
    // print_seir_params(model);
    auto integrator = std::make_shared<mio::RKIntegratorCore<double>>();
    integrator->set_dt_max(dt);
    integrator->set_abs_tolerance(1e-6);
    integrator->set_rel_tolerance(1e-6);

    auto seir = mio::simulate<mio::oseir::Model<double>,double>(t0, tmax, dt, model, integrator);
    mio::time_series_to_file(seir,"seir.txt");

    printf("\n number total: %f\n",
           seir.get_last_value()[0] + seir.get_last_value()[1] + seir.get_last_value()[2] + seir.get_last_value()[3]);
}