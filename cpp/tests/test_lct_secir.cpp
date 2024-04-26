/* 
* Copyright (C) 2020-2024 MEmilio
*
* Authors: Lena Ploetzke
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

#include "lct_secir/model.h"
#include "lct_secir/infection_state.h"
#include "lct_secir/simulation.h"
#include "lct_secir/parameters.h"
#include "ode_secir/model.h"
#include "memilio/config.h"
#include "memilio/utils/time_series.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/math/eigen.h"
#include "load_test_data.h"

#include <gtest/gtest.h>
#include "boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp"

// Test confirms that default construction of an LCT model works.
TEST(TestLCTSecir, simulateDefault)
{
    using Model     = mio::lsecir::Model<1, 1, 1, 1, 1>;
    ScalarType t0   = 0;
    ScalarType tmax = 1;
    ScalarType dt   = 0.1;

    Eigen::VectorXd init = Eigen::VectorXd::Constant((int)Model::LctState::InfectionState::Count, 15);
    init[0]              = 200;
    init[3]              = 50;
    init[5]              = 30;

    Model model(init);
    mio::TimeSeries<ScalarType> result = mio::lsecir::simulate(t0, tmax, dt, model);

    EXPECT_NEAR(result.get_last_time(), tmax, 1e-10);
    ScalarType sum_pop = init.sum();
    for (Eigen::Index i = 0; i < result.get_num_time_points(); i++) {
        ASSERT_NEAR(sum_pop, result[i].sum(), 1e-5);
    }
}

/* Test compares the result for an LCT SECIR model with one single subcompartment for each infection state 
    with the result of the equivalent ODE SECIR model. */
TEST(TestLCTSecir, compareWithOdeSecir)
{
    using Model     = mio::lsecir::Model<1, 1, 1, 1, 1>;
    ScalarType t0   = 0;
    ScalarType tmax = 5;
    ScalarType dt   = 0.1;

    // Initialization vector for both models.
    Eigen::VectorXd init = Eigen::VectorXd::Constant((int)Model::LctState::InfectionState::Count, 15);
    init[0]              = 200;
    init[3]              = 50;
    init[5]              = 30;

    // Define LCT model.
    Model model_lct(init);
    // Set Parameters.
    model_lct.parameters.get<mio::lsecir::TimeExposed>()            = 3.2;
    model_lct.parameters.get<mio::lsecir::TimeInfectedNoSymptoms>() = 2;
    model_lct.parameters.get<mio::lsecir::TimeInfectedSymptoms>()   = 5.8;
    model_lct.parameters.get<mio::lsecir::TimeInfectedSevere>()     = 9.5;
    model_lct.parameters.get<mio::lsecir::TimeInfectedCritical>()   = 7.1;

    model_lct.parameters.get<mio::lsecir::TransmissionProbabilityOnContact>() = 0.05;

    mio::ContactMatrixGroup<>& contact_matrix_lct = model_lct.parameters.get<mio::lsecir::ContactPatterns>();
    contact_matrix_lct[0]                         = mio::ContactMatrix<>(Eigen::MatrixXd::Constant(1, 1, 10));
    contact_matrix_lct[0].add_damping(0.7, mio::SimulationTime<>(2.));

    model_lct.parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 0.7;
    model_lct.parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 0.25;
    model_lct.parameters.get<mio::lsecir::StartDay>()                       = 50;
    model_lct.parameters.get<mio::lsecir::Seasonality>()                    = 0.1;
    model_lct.parameters.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 0.09;
    model_lct.parameters.get<mio::lsecir::SeverePerInfectedSymptoms>()      = 0.2;
    model_lct.parameters.get<mio::lsecir::CriticalPerSevere>()              = 0.25;
    model_lct.parameters.get<mio::lsecir::DeathsPerCritical>()              = 0.3;

    // Simulate.
    mio::TimeSeries<ScalarType> result_lct = mio::lsecir::simulate(
        t0, tmax, dt, model_lct,
        std::make_shared<mio::ControlledStepperWrapper<ScalarType, boost::numeric::odeint::runge_kutta_cash_karp54>>());

    // Initialize ODE model with one age group.
    mio::osecir::Model model_ode(1);
    // Set initial distribution of the population.
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}] =
        init[Eigen::Index(Model::LctState::InfectionState::Exposed)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] =
        init[Eigen::Index(Model::LctState::InfectionState::InfectedNoSymptoms)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptomsConfirmed}] = 0;
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedSymptoms}] =
        init[Eigen::Index(Model::LctState::InfectionState::InfectedSymptoms)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedSymptomsConfirmed}] = 0;
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedSevere}] =
        init[Eigen::Index(Model::LctState::InfectionState::InfectedSevere)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedCritical}] =
        init[Eigen::Index(Model::LctState::InfectionState::InfectedCritical)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Recovered}] =
        init[Eigen::Index(Model::LctState::InfectionState::Recovered)];
    model_ode.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Dead}] =
        init[Eigen::Index(Model::LctState::InfectionState::Dead)];
    model_ode.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                                    init.sum());

    // Set parameters according to the parameters of the LCT model.
    // No restrictions by additional parameters.
    model_ode.parameters.get<mio::osecir::TestAndTraceCapacity<double>>() = std::numeric_limits<double>::max();
    model_ode.parameters.get<mio::osecir::ICUCapacity<double>>()          = std::numeric_limits<double>::max();

    model_ode.parameters.set<mio::osecir::StartDay>(50);
    model_ode.parameters.set<mio::osecir::Seasonality<double>>(0.1);
    model_ode.parameters.get<mio::osecir::TimeExposed<double>>()[(mio::AgeGroup)0]            = 3.2;
    model_ode.parameters.get<mio::osecir::TimeInfectedNoSymptoms<double>>()[(mio::AgeGroup)0] = 2.0;
    model_ode.parameters.get<mio::osecir::TimeInfectedSymptoms<double>>()[(mio::AgeGroup)0]   = 5.8;
    model_ode.parameters.get<mio::osecir::TimeInfectedSevere<double>>()[(mio::AgeGroup)0]     = 9.5;
    model_ode.parameters.get<mio::osecir::TimeInfectedCritical<double>>()[(mio::AgeGroup)0]   = 7.1;

    mio::ContactMatrixGroup<>& contact_matrix_ode = model_ode.parameters.get<mio::osecir::ContactPatterns<double>>();
    contact_matrix_ode[0]                         = mio::ContactMatrix<>(Eigen::MatrixXd::Constant(1, 1, 10));
    contact_matrix_ode[0].add_damping(0.7, mio::SimulationTime<>(2.));

    model_ode.parameters.get<mio::osecir::TransmissionProbabilityOnContact<double>>()[(mio::AgeGroup)0] = 0.05;
    model_ode.parameters.get<mio::osecir::RelativeTransmissionNoSymptoms<double>>()[(mio::AgeGroup)0]   = 0.7;
    model_ode.parameters.get<mio::osecir::RecoveredPerInfectedNoSymptoms<double>>()[(mio::AgeGroup)0]   = 0.09;
    model_ode.parameters.get<mio::osecir::RiskOfInfectionFromSymptomatic<double>>()[(mio::AgeGroup)0]   = 0.25;
    model_ode.parameters.get<mio::osecir::SeverePerInfectedSymptoms<double>>()[(mio::AgeGroup)0]        = 0.2;
    model_ode.parameters.get<mio::osecir::CriticalPerSevere<double>>()[(mio::AgeGroup)0]                = 0.25;
    model_ode.parameters.get<mio::osecir::DeathsPerCritical<double>>()[(mio::AgeGroup)0]                = 0.3;

    // Simulate.
    mio::TimeSeries<double> result_ode = mio::osecir::simulate<double>(
        t0, tmax, dt, model_ode,
        std::make_shared<mio::ControlledStepperWrapper<double, boost::numeric::odeint::runge_kutta_cash_karp54>>());

    // Simulation results should be equal.
    ASSERT_EQ(result_lct.get_num_time_points(), result_ode.get_num_time_points());
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(result_lct.get_time(i), result_ode.get_time(i), 1e-5);

        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::Susceptible],
                    result_ode[i][(int)mio::osecir::InfectionState::Susceptible], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::Exposed],
                    result_ode[i][(int)mio::osecir::InfectionState::Exposed], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::InfectedNoSymptoms],
                    result_ode[i][(int)mio::osecir::InfectionState::InfectedNoSymptoms], 1e-5);
        ASSERT_NEAR(0, result_ode[i][(int)mio::osecir::InfectionState::InfectedNoSymptomsConfirmed], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::InfectedSymptoms],
                    result_ode[i][(int)mio::osecir::InfectionState::InfectedSymptoms], 1e-5);
        ASSERT_NEAR(0, result_ode[i][(int)mio::osecir::InfectionState::InfectedSymptomsConfirmed], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::InfectedCritical],
                    result_ode[i][(int)mio::osecir::InfectionState::InfectedCritical], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::InfectedSevere],
                    result_ode[i][(int)mio::osecir::InfectionState::InfectedSevere], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::Recovered],
                    result_ode[i][(int)mio::osecir::InfectionState::Recovered], 1e-5);
        ASSERT_NEAR(result_lct[i][(int)Model::LctState::InfectionState::Dead],
                    result_ode[i][(int)mio::osecir::InfectionState::Dead], 1e-5);
    }
}

// Test if the function eval_right_hand_side() is working using a hand calculated result.
TEST(TestLCTSecir, testEvalRightHandSide)
{
    // Define model.
    // Chose more than one subcompartment for all compartments except S, R, D so that the function is correct for all selections.
    using Model    = mio::lsecir::Model<2, 3, 2, 2, 2>;
    using LctState = Model::LctState;

    // Define initial population distribution in infection states, one entry per subcompartment.
    Eigen::VectorXd init(LctState::Count);
    init[LctState::get_first_index<LctState::InfectionState::Susceptible>()]            = 750;
    init[LctState::get_first_index<LctState::InfectionState::Exposed>()]                = 30;
    init[LctState::get_first_index<LctState::InfectionState::Exposed>() + 1]            = 20;
    init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>()]     = 20;
    init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>() + 1] = 10;
    init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>() + 2] = 10;
    init[LctState::get_first_index<LctState::InfectionState::InfectedSymptoms>()]       = 30;
    init[LctState::get_first_index<LctState::InfectionState::InfectedSymptoms>() + 1]   = 20;
    init[LctState::get_first_index<LctState::InfectionState::InfectedSevere>()]         = 40;
    init[LctState::get_first_index<LctState::InfectionState::InfectedSevere>() + 1]     = 10;
    init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>()]       = 10;
    init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>() + 1]   = 20;
    init[LctState::get_first_index<LctState::InfectionState::Recovered>()]              = 20;
    init[LctState::get_first_index<LctState::InfectionState::Dead>()]                   = 10;

    Model model(std::move(init));

    // Set parameters.
    model.parameters.set<mio::lsecir::TimeExposed>(3.2);
    model.parameters.get<mio::lsecir::TimeInfectedNoSymptoms>() = 2;
    model.parameters.get<mio::lsecir::TimeInfectedSymptoms>()   = 5.8;
    model.parameters.get<mio::lsecir::TimeInfectedSevere>()     = 9.5;
    model.parameters.get<mio::lsecir::TimeInfectedCritical>()   = 7.1;

    model.parameters.get<mio::lsecir::TransmissionProbabilityOnContact>() = 0.05;

    mio::ContactMatrixGroup<>& contact_matrix = model.parameters.get<mio::lsecir::ContactPatterns>();
    contact_matrix[0]                         = mio::ContactMatrix<>(Eigen::MatrixXd::Constant(1, 1, 10));

    model.parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 0.7;
    model.parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 0.25;
    model.parameters.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 0.09;
    model.parameters.get<mio::lsecir::Seasonality>()                    = 0.;
    model.parameters.get<mio::lsecir::StartDay>()                       = 0;
    model.parameters.get<mio::lsecir::SeverePerInfectedSymptoms>()      = 0.2;
    model.parameters.get<mio::lsecir::CriticalPerSevere>()              = 0.25;
    model.parameters.get<mio::lsecir::DeathsPerCritical>()              = 0.3;

    // Compare the result of eval_right_hand_side() with a hand calculated result.
    unsigned int num_subcompartments = LctState::Count;
    Eigen::VectorXd dydt(num_subcompartments);
    model.eval_right_hand_side(model.get_initial_values(), 0, dydt);

    Eigen::VectorXd compare(num_subcompartments);
    compare << -15.3409, -3.4091, 6.25, -17.5, 15, 0, 3.3052, 3.4483, -7.0417, 6.3158, -2.2906, -2.8169, 12.3899,
        1.6901;

    for (unsigned int i = 0; i < num_subcompartments; i++) {
        ASSERT_NEAR(compare[i], dydt[i], 1e-3);
    }
}

// Model setup to compare result with a previous output.
class ModelTestLCTSecir : public testing::Test
{
public:
    using Model    = mio::lsecir::Model<2, 3, 1, 1, 5>;
    using LctState = Model::LctState;

protected:
    virtual void SetUp()
    {
        // Define initial distribution of the population in the subcompartments.
        Eigen::VectorXd init(LctState::Count);
        init[LctState::get_first_index<LctState::InfectionState::Susceptible>()]            = 750;
        init[LctState::get_first_index<LctState::InfectionState::Exposed>()]                = 30;
        init[LctState::get_first_index<LctState::InfectionState::Exposed>() + 1]            = 20;
        init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>()]     = 20;
        init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>() + 1] = 10;
        init[LctState::get_first_index<LctState::InfectionState::InfectedNoSymptoms>() + 2] = 10;
        init[LctState::get_first_index<LctState::InfectionState::InfectedSymptoms>()]       = 50;
        init[LctState::get_first_index<LctState::InfectionState::InfectedSevere>()]         = 50;
        init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>()]       = 10;
        init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>() + 1]   = 10;
        init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>() + 2]   = 5;
        init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>() + 3]   = 3;
        init[LctState::get_first_index<LctState::InfectionState::InfectedCritical>() + 4]   = 2;
        init[LctState::get_first_index<LctState::InfectionState::Recovered>()]              = 20;
        init[LctState::get_first_index<LctState::InfectionState::Dead>()]                   = 10;

        // Initialize model and set parameters.
        model                                                                  = new Model(std::move(init));
        model->parameters.get<mio::lsecir::TimeExposed>()                      = 3.2;
        model->parameters.get<mio::lsecir::TimeInfectedNoSymptoms>()           = 2;
        model->parameters.get<mio::lsecir::TimeInfectedSymptoms>()             = 5.8;
        model->parameters.get<mio::lsecir::TimeInfectedSevere>()               = 9.5;
        model->parameters.get<mio::lsecir::TimeInfectedCritical>()             = 7.1;
        model->parameters.get<mio::lsecir::TransmissionProbabilityOnContact>() = 0.05;

        mio::ContactMatrixGroup<>& contact_matrix = model->parameters.get<mio::lsecir::ContactPatterns>();
        contact_matrix[0]                         = mio::ContactMatrix<>(Eigen::MatrixXd::Constant(1, 1, 10));
        contact_matrix[0].add_damping(0.7, mio::SimulationTime<>(2.));

        model->parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 0.7;
        model->parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 0.25;
        model->parameters.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 0.09;
        model->parameters.get<mio::lsecir::SeverePerInfectedSymptoms>()      = 0.2;
        model->parameters.get<mio::lsecir::CriticalPerSevere>()              = 0.25;
        model->parameters.get<mio::lsecir::DeathsPerCritical>()              = 0.3;
    }

    virtual void TearDown()
    {
        delete model;
    }

public:
    Model* model = nullptr;
};

// Test compares a simulation with the result of a previous run stored in a .csv file.
TEST_F(ModelTestLCTSecir, compareWithPreviousRun)
{
    ScalarType tmax                    = 3;
    mio::TimeSeries<ScalarType> result = mio::lsecir::simulate(
        0, tmax, 0.5, *model,
        std::make_shared<mio::ControlledStepperWrapper<ScalarType, boost::numeric::odeint::runge_kutta_cash_karp54>>());

    // Compare subcompartments.
    auto compare = load_test_data_csv<ScalarType>("lct-secir-subcompartments-compare.csv");

    ASSERT_EQ(compare.size(), static_cast<size_t>(result.get_num_time_points()));
    for (size_t i = 0; i < compare.size(); i++) {
        ASSERT_EQ(compare[i].size(), static_cast<size_t>(result.get_num_elements()) + 1) << "at row " << i;
        ASSERT_NEAR(result.get_time(i), compare[i][0], 1e-7) << "at row " << i;
        for (size_t j = 1; j < compare[i].size(); j++) {
            ASSERT_NEAR(result.get_value(i)[j - 1], compare[i][j], 1e-7) << " at row " << i;
        }
    }

    // Compare InfectionState compartments.
    mio::TimeSeries<ScalarType> population = model->calculate_populations(result);
    auto compare_population                = load_test_data_csv<ScalarType>("lct-secir-compartments-compare.csv");

    ASSERT_EQ(compare_population.size(), static_cast<size_t>(population.get_num_time_points()));
    for (size_t i = 0; i < compare_population.size(); i++) {
        ASSERT_EQ(compare_population[i].size(), static_cast<size_t>(population.get_num_elements()) + 1)
            << "at row " << i;
        ASSERT_NEAR(population.get_time(i), compare_population[i][0], 1e-7) << "at row " << i;
        for (size_t j = 1; j < compare_population[i].size(); j++) {
            ASSERT_NEAR(population.get_value(i)[j - 1], compare_population[i][j], 1e-7) << " at row " << i;
        }
    }
}

// Test calculate_populations with a vector of a wrong size.
TEST_F(ModelTestLCTSecir, testCalculatePopWrongSize)
{
    // Deactivate temporarily log output because an error is expected.
    mio::set_log_level(mio::LogLevel::off);
    mio::TimeSeries<ScalarType> init_wrong_size((int)LctState::InfectionState::Count);
    Eigen::VectorXd vec_wrong_size = Eigen::VectorXd::Ones((int)LctState::InfectionState::Count);
    init_wrong_size.add_time_point(-10, vec_wrong_size);
    init_wrong_size.add_time_point(-9, vec_wrong_size);
    mio::TimeSeries<ScalarType> population = model->calculate_populations(init_wrong_size);
    EXPECT_EQ(1, population.get_num_time_points());
    for (int i = 0; i < population.get_num_elements(); i++) {
        EXPECT_EQ(-1, population.get_last_value()[i]);
    }
    // Reactive log output.
    mio::set_log_level(mio::LogLevel::warn);
}

// Check constraints of Parameters and Model.
TEST(TestLCTSecir, testConstraints)
{
    // Deactivate temporarily log output for next tests.
    mio::set_log_level(mio::LogLevel::off);

    // Check for exceptions of parameters.
    mio::lsecir::Parameters parameters_lct;
    parameters_lct.get<mio::lsecir::TimeExposed>()                      = 0;
    parameters_lct.get<mio::lsecir::TimeInfectedNoSymptoms>()           = 3.1;
    parameters_lct.get<mio::lsecir::TimeInfectedSymptoms>()             = 6.1;
    parameters_lct.get<mio::lsecir::TimeInfectedSevere>()               = 11.1;
    parameters_lct.get<mio::lsecir::TimeInfectedCritical>()             = 17.1;
    parameters_lct.get<mio::lsecir::TransmissionProbabilityOnContact>() = 0.01;
    mio::ContactMatrixGroup<> contact_matrix                            = mio::ContactMatrixGroup<>(1, 1);
    parameters_lct.get<mio::lsecir::ContactPatterns>()                  = mio::UncertainContactMatrix(contact_matrix);

    parameters_lct.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 1;
    parameters_lct.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 1;
    parameters_lct.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 0.1;
    parameters_lct.get<mio::lsecir::SeverePerInfectedSymptoms>()      = 0.1;
    parameters_lct.get<mio::lsecir::CriticalPerSevere>()              = 0.1;
    parameters_lct.get<mio::lsecir::DeathsPerCritical>()              = 0.1;

    // Check improper TimeExposed.
    bool constraint_check = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TimeExposed>() = 3.1;

    // Check TimeInfectedNoSymptoms.
    parameters_lct.get<mio::lsecir::TimeInfectedNoSymptoms>() = 0.1;
    constraint_check                                          = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TimeInfectedNoSymptoms>() = 3.1;

    // Check TimeInfectedSymptoms.
    parameters_lct.get<mio::lsecir::TimeInfectedSymptoms>() = -0.1;
    constraint_check                                        = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TimeInfectedSymptoms>() = 6.1;

    // Check TimeInfectedSevere.
    parameters_lct.get<mio::lsecir::TimeInfectedSevere>() = 0.5;
    constraint_check                                      = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TimeInfectedSevere>() = 11.1;

    // Check TimeInfectedCritical.
    parameters_lct.get<mio::lsecir::TimeInfectedCritical>() = 0.;
    constraint_check                                        = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TimeInfectedCritical>() = 17.1;

    // Check TransmissionProbabilityOnContact.
    parameters_lct.get<mio::lsecir::TransmissionProbabilityOnContact>() = -1;
    constraint_check                                                    = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::TransmissionProbabilityOnContact>() = 0.01;

    // Check RelativeTransmissionNoSymptoms.
    parameters_lct.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 1.5;
    constraint_check                                                  = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::RelativeTransmissionNoSymptoms>() = 1;

    // Check RiskOfInfectionFromSymptomatic.
    parameters_lct.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 1.5;
    constraint_check                                                  = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::RiskOfInfectionFromSymptomatic>() = 1;

    // Check RecoveredPerInfectedNoSymptoms.
    parameters_lct.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 1.5;
    constraint_check                                                  = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>() = 0.1;

    // Check SeverePerInfectedSymptoms.
    parameters_lct.get<mio::lsecir::SeverePerInfectedSymptoms>() = -1;
    constraint_check                                             = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::SeverePerInfectedSymptoms>() = 0.1;

    // Check CriticalPerSevere.
    parameters_lct.get<mio::lsecir::CriticalPerSevere>() = -1;
    constraint_check                                     = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::CriticalPerSevere>() = 0.1;

    // Check DeathsPerCritical.
    parameters_lct.get<mio::lsecir::DeathsPerCritical>() = -1;
    constraint_check                                     = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::DeathsPerCritical>() = 0.1;

    // Check Seasonality.
    parameters_lct.get<mio::lsecir::Seasonality>() = 1;
    constraint_check                               = parameters_lct.check_constraints();
    EXPECT_TRUE(constraint_check);
    parameters_lct.get<mio::lsecir::Seasonality>() = 0.1;

    // Check with correct parameters.
    constraint_check = parameters_lct.check_constraints();
    EXPECT_FALSE(constraint_check);

    // Check for model.
    using Model    = mio::lsecir::Model<1, 1, 1, 1, 1>;
    using LctState = Model::LctState;

    // Check wrong size of initial value vector.
    Model model1(std::move(Eigen::VectorXd::Ones((int)LctState::Count - 1)), std::move(parameters_lct));
    constraint_check = model1.check_constraints();
    EXPECT_TRUE(constraint_check);

    // Check with values smaller than zero.
    Model model2(std::move(Eigen::VectorXd::Constant((int)LctState::Count, -1)), std::move(parameters_lct));
    constraint_check = model2.check_constraints();
    EXPECT_TRUE(constraint_check);

    // Check with correct conditions.
    Model model3(std::move(Eigen::VectorXd::Constant((int)LctState::Count, 1)));
    constraint_check = model3.check_constraints();
    EXPECT_FALSE(constraint_check);

    // Reactive log output.
    mio::set_log_level(mio::LogLevel::warn);
}
