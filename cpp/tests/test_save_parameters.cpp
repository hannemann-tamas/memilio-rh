#include "load_test_data.h"
#include "epidemiology/secir.h"
#include <epidemiology_io/secir_result_io.h>
#include <epidemiology_io/secir_parameters_io.h>
#include <distributions_helpers.h>
// #include <epidemiology/parameter_studies/parameter_studies.h>
#include <gtest/gtest.h>

TEST(TestSaveParameters, compareParameterStudy)
{
    double t0   = 0.0;
    double tmax = 50.5;
    double dt   = 0.1;

    double tinc = 5.2, tinfmild = 6, tserint = 4.2, thosp2home = 12, thome2hosp = 5, thosp2icu = 2, ticu2home = 8,
           tinfasy = 6.2, ticu2death = 5;

    double cont_freq = 0.5, alpha = 0.09, beta = 0.25, delta = 0.3, rho = 0.2, theta = 0.25;

    double num_total_t0 = 10000, num_exp_t0 = 100, num_inf_t0 = 50, num_car_t0 = 50, num_hosp_t0 = 20, num_icu_t0 = 10,
           num_rec_t0 = 10, num_dead_t0 = 0;

    int num_groups = 2;
    double fact    = 1.0 / (double)num_groups;

    epi::SecirParams params(num_groups);

    for (size_t i = 0; i < num_groups; i++) {
        params.times[i].set_incubation(tinc);
        params.times[i].set_infectious_mild(tinfmild);
        params.times[i].set_serialinterval(tserint);
        params.times[i].set_hospitalized_to_home(thosp2home);
        params.times[i].set_home_to_hospitalized(thome2hosp);
        params.times[i].set_hospitalized_to_icu(thosp2icu);
        params.times[i].set_icu_to_home(ticu2home);
        params.times[i].set_infectious_asymp(tinfasy);
        params.times[i].set_icu_to_death(ticu2death);

        params.populations.set({i, epi::SecirCompartments::E}, fact * num_exp_t0);
        params.populations.set({i, epi::SecirCompartments::C}, fact * num_car_t0);
        params.populations.set({i, epi::SecirCompartments::I}, fact * num_inf_t0);
        params.populations.set({i, epi::SecirCompartments::H}, fact * num_hosp_t0);
        params.populations.set({i, epi::SecirCompartments::U}, fact * num_icu_t0);
        params.populations.set({i, epi::SecirCompartments::R}, fact * num_rec_t0);
        params.populations.set({i, epi::SecirCompartments::D}, fact * num_dead_t0);
        params.populations.set_difference_from_group_total({i, epi::SecirCompartments::S}, epi::SecirCategory::AgeGroup,
                                                           i, fact * num_total_t0);

        params.probabilities[i].set_asymp_per_infectious(alpha);
        params.probabilities[i].set_risk_from_symptomatic(beta);
        params.probabilities[i].set_hospitalized_per_infectious(rho);
        params.probabilities[i].set_icu_per_hospitalized(theta);
        params.probabilities[i].set_dead_per_icu(delta);
    }

    epi::ContactFrequencyMatrix& cont_freq_matrix = params.get_contact_patterns();
    epi::Damping dummy(30., 0.3);
    for (int i = 0; i < num_groups; i++) {
        for (int j = i; j < num_groups; j++) {
            cont_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
            cont_freq_matrix.add_damping(dummy, i, j);
        }
    }
    cont_freq_matrix.add_damping(epi::Damping{35, 0.2}, 0, 0);
    int num_runs     = 5;
    std::string path = "/Parameters";
    TixiDocumentHandle handle;

    tixiCreateDocument("Parameters", &handle);
    epi::ParameterStudy study(
        [](auto&& t0, auto&& tmax, auto&& dt, auto&& params) {
            return epi::simulate(t0, tmax, dt, params);
        },
        params, t0, tmax, 0.0, num_runs);

    study.get_parameter_space().get_secir_params().times[0].get_incubation().get_distribution()->add_predefined_sample(
        4711.0);

    study.get_parameter_space()
        .get_secir_params()
        .get_contact_patterns()
        .get_distribution_damp_days()
        ->add_predefined_sample(4711.0);

    epi::write_parameter_study(handle, path, study);
    tixiSaveDocument(handle, "TestParameters.xml");
    tixiCloseDocument(handle);

    tixiOpenDocument("TestParameters.xml", &handle);
    epi::ParameterStudy read_study = epi::read_parameter_study(handle, path);
    tixiCloseDocument(handle);

    ASSERT_EQ(study.get_num_runs(), read_study.get_num_runs());
    ASSERT_EQ(study.get_t0(), read_study.get_t0());
    ASSERT_EQ(study.get_tmax(), read_study.get_tmax());

    const epi::ParameterSpace& space      = study.get_parameter_space();
    const epi::ParameterSpace& read_space = read_study.get_parameter_space();

    const epi::UncertainContactMatrix& contact      = space.get_secir_params().get_contact_patterns();
    const epi::UncertainContactMatrix& read_contact = read_space.get_secir_params().get_contact_patterns();

    num_groups          = space.get_secir_params().get_num_groups();
    int num_groups_read = read_space.get_secir_params().get_num_groups();
    ASSERT_EQ(num_groups, num_groups_read);

    for (int i = 0; i < num_groups; i++) {
        ASSERT_EQ(space.get_dead(i), read_space.get_dead(i));
        ASSERT_EQ(space.get_total(i), read_space.get_total(i));

        check_distribution(*space.get_distribution_exposed(i), *read_space.get_distribution_exposed(i));
        check_distribution(*space.get_distribution_carrier(i), *read_space.get_distribution_carrier(i));
        check_distribution(*space.get_distribution_infectious(i), *read_space.get_distribution_infectious(i));
        check_distribution(*space.get_distribution_hospitalized(i), *read_space.get_distribution_hospitalized(i));
        check_distribution(*space.get_distribution_icu(i), *read_space.get_distribution_icu(i));
        check_distribution(*space.get_distribution_recovered(i), *read_space.get_distribution_recovered(i));

        check_distribution(*space.get_distribution_incubation(i), *read_space.get_distribution_incubation(i));
        check_distribution(*space.get_distribution_inf_mild(i), *read_space.get_distribution_inf_mild(i));
        check_distribution(*space.get_distribution_serial_int(i), *read_space.get_distribution_serial_int(i));
        check_distribution(*space.get_distribution_hosp_to_rec(i), *read_space.get_distribution_hosp_to_rec(i));
        check_distribution(*space.get_distribution_inf_to_hosp(i), *read_space.get_distribution_inf_to_hosp(i));
        check_distribution(*space.get_distribution_inf_asymp(i), *read_space.get_distribution_inf_asymp(i));
        check_distribution(*space.get_distribution_hosp_to_icu(i), *read_space.get_distribution_hosp_to_icu(i));
        check_distribution(*space.get_distribution_icu_to_rec(i), *read_space.get_distribution_icu_to_rec(i));
        check_distribution(*space.get_distribution_icu_to_death(i), *read_space.get_distribution_icu_to_death(i));

        check_distribution(*space.get_distribution_inf_from_cont(i), *read_space.get_distribution_inf_from_cont(i));
        check_distribution(*space.get_distribution_risk_from_symp(i), *read_space.get_distribution_risk_from_symp(i));
        check_distribution(*space.get_distribution_asymp_per_inf(i), *read_space.get_distribution_asymp_per_inf(i));
        check_distribution(*space.get_distribution_death_per_icu(i), *read_space.get_distribution_death_per_icu(i));
        check_distribution(*space.get_distribution_hosp_per_inf(i), *read_space.get_distribution_hosp_per_inf(i));
        check_distribution(*space.get_distribution_icu_per_hosp(i), *read_space.get_distribution_icu_per_hosp(i));

        for (int j = 0; j < num_groups; j++) {
            ASSERT_EQ(contact.get_cont_freq_mat().get_cont_freq(i, j),
                      read_contact.get_cont_freq_mat().get_cont_freq(i, j));

            ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size(),
                      read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size());
            for (int k = 0; k < contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size(); k++) {
                ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].day,
                          read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].day);
                ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].factor,
                          read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].factor);
            }
        }

        check_distribution(*contact.get_distribution_damp_nb().get(), *read_contact.get_distribution_damp_nb().get());
        check_distribution(*contact.get_distribution_damp_days().get(),
                           *read_contact.get_distribution_damp_days().get());
        check_distribution(*contact.get_distribution_damp_diag_base().get(),
                           *read_contact.get_distribution_damp_diag_base().get());
        check_distribution(*contact.get_distribution_damp_diag_rel().get(),
                           *read_contact.get_distribution_damp_diag_rel().get());
        check_distribution(*contact.get_distribution_damp_offdiag_rel().get(),
                           *read_contact.get_distribution_damp_offdiag_rel().get());
    }
}

TEST(TestSaveParameters, compareSingleRun)
{
    double t0   = 0.0;
    double tmax = 50.5;
    double dt   = 0.1;

    double tinc = 5.2, tinfmild = 6, tserint = 4.2, thosp2home = 12, thome2hosp = 5, thosp2icu = 2, ticu2home = 8,
           tinfasy = 6.2, ticu2death = 5;

    double cont_freq = 0.5, alpha = 0.09, beta = 0.25, delta = 0.3, rho = 0.2, theta = 0.25;

    double num_total_t0 = 10000, num_exp_t0 = 100, num_inf_t0 = 50, num_car_t0 = 50, num_hosp_t0 = 20, num_icu_t0 = 10,
           num_rec_t0 = 10, num_dead_t0 = 0;

    int num_groups = 2;
    double fact    = 1.0 / (double)num_groups;

    epi::SecirParams params(num_groups);

    for (size_t i = 0; i < num_groups; i++) {
        params.times[i].set_incubation(tinc);
        params.times[i].set_infectious_mild(tinfmild);
        params.times[i].set_serialinterval(tserint);
        params.times[i].set_hospitalized_to_home(thosp2home);
        params.times[i].set_home_to_hospitalized(thome2hosp);
        params.times[i].set_hospitalized_to_icu(thosp2icu);
        params.times[i].set_icu_to_home(ticu2home);
        params.times[i].set_infectious_asymp(tinfasy);
        params.times[i].set_icu_to_death(ticu2death);

        params.populations.set({i, epi::SecirCompartments::E}, fact * num_exp_t0);
        params.populations.set({i, epi::SecirCompartments::C}, fact * num_car_t0);
        params.populations.set({i, epi::SecirCompartments::I}, fact * num_inf_t0);
        params.populations.set({i, epi::SecirCompartments::H}, fact * num_hosp_t0);
        params.populations.set({i, epi::SecirCompartments::U}, fact * num_icu_t0);
        params.populations.set({i, epi::SecirCompartments::R}, fact * num_rec_t0);
        params.populations.set({i, epi::SecirCompartments::D}, fact * num_dead_t0);
        params.populations.set_difference_from_group_total({i, epi::SecirCompartments::S}, epi::SecirCategory::AgeGroup,
                                                           i, fact * num_total_t0);

        params.probabilities[i].set_asymp_per_infectious(alpha);
        params.probabilities[i].set_risk_from_symptomatic(beta);
        params.probabilities[i].set_hospitalized_per_infectious(rho);
        params.probabilities[i].set_icu_per_hospitalized(theta);
        params.probabilities[i].set_dead_per_icu(delta);
    }

    epi::ContactFrequencyMatrix& cont_freq_matrix = params.get_contact_patterns();
    epi::Damping dummy(30., 0.3);
    for (int i = 0; i < num_groups; i++) {
        for (int j = i; j < num_groups; j++) {
            cont_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
            cont_freq_matrix.add_damping(dummy, i, j);
        }
    }
    cont_freq_matrix.add_damping(epi::Damping{35, 0.2}, 0, 0);
    int num_runs     = 5;
    std::string path = "/Parameters";
    TixiDocumentHandle handle;

    tixiCreateDocument("Parameters", &handle);
    epi::ParameterStudy study(
        [](auto&& t0, auto&& tmax, auto&& dt, auto&& params) {
            return epi::simulate(t0, tmax, dt, params);
        },
        params, t0, tmax, 0.0, num_runs);

    epi::write_parameter_study(handle, path, study, 0);
    tixiSaveDocument(handle, "TestParameterValues.xml");
    tixiCloseDocument(handle);

    tixiOpenDocument("TestParameterValues.xml", &handle);
    epi::ParameterStudy read_study = epi::read_parameter_study(handle, path);
    tixiCloseDocument(handle);

    ASSERT_EQ(study.get_num_runs(), read_study.get_num_runs());
    ASSERT_EQ(study.get_t0(), read_study.get_t0());
    ASSERT_EQ(study.get_tmax(), read_study.get_tmax());

    const epi::ParameterSpace& space      = study.get_parameter_space();
    const epi::ParameterSpace& read_space = read_study.get_parameter_space();

    const epi::SecirParams& read_params = read_space.get_secir_params();

    const epi::UncertainContactMatrix& contact      = space.get_secir_params().get_contact_patterns();
    const epi::UncertainContactMatrix& read_contact = read_space.get_secir_params().get_contact_patterns();

    num_groups          = space.get_secir_params().get_num_groups();
    int num_groups_read = read_space.get_secir_params().get_num_groups();
    ASSERT_EQ(num_groups, num_groups_read);

    for (size_t i = 0; i < num_groups; i++) {
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::D}),
                  read_params.populations.get({i, epi::SecirCompartments::D}));
        ASSERT_EQ(params.populations.get_group_total(epi::AgeGroup, i),
                  read_params.populations.get_group_total(epi::AgeGroup, i));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::E}),
                  read_params.populations.get({i, epi::SecirCompartments::E}));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::C}),
                  read_params.populations.get({i, epi::SecirCompartments::C}));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::I}),
                  read_params.populations.get({i, epi::SecirCompartments::I}));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::H}),
                  read_params.populations.get({i, epi::SecirCompartments::H}));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::U}),
                  read_params.populations.get({i, epi::SecirCompartments::U}));
        ASSERT_EQ(params.populations.get({i, epi::SecirCompartments::R}),
                  read_params.populations.get({i, epi::SecirCompartments::R}));

        ASSERT_EQ(params.times[i].get_incubation(), read_params.times[i].get_incubation());
        ASSERT_EQ(params.times[i].get_infectious_mild(), read_params.times[i].get_infectious_mild());
        ASSERT_EQ(params.times[i].get_serialinterval(), read_params.times[i].get_serialinterval());
        ASSERT_EQ(params.times[i].get_hospitalized_to_home(), read_params.times[i].get_hospitalized_to_home());
        ASSERT_EQ(params.times[i].get_home_to_hospitalized(), read_params.times[i].get_home_to_hospitalized());
        ASSERT_EQ(params.times[i].get_infectious_asymp(), read_params.times[i].get_infectious_asymp());
        ASSERT_EQ(params.times[i].get_hospitalized_to_icu(), read_params.times[i].get_hospitalized_to_icu());
        ASSERT_EQ(params.times[i].get_icu_to_home(), read_params.times[i].get_icu_to_home());
        ASSERT_EQ(params.times[i].get_icu_to_dead(), read_params.times[i].get_icu_to_dead());

        ASSERT_EQ(params.probabilities[i].get_infection_from_contact(),
                  read_params.probabilities[i].get_infection_from_contact());
        ASSERT_EQ(params.probabilities[i].get_risk_from_symptomatic(),
                  read_params.probabilities[i].get_risk_from_symptomatic());
        ASSERT_EQ(params.probabilities[i].get_asymp_per_infectious(),
                  read_params.probabilities[i].get_asymp_per_infectious());
        ASSERT_EQ(params.probabilities[i].get_dead_per_icu(), read_params.probabilities[i].get_dead_per_icu());
        ASSERT_EQ(params.probabilities[i].get_hospitalized_per_infectious(),
                  read_params.probabilities[i].get_hospitalized_per_infectious());
        ASSERT_EQ(params.probabilities[i].get_icu_per_hospitalized(),
                  read_params.probabilities[i].get_icu_per_hospitalized());

        for (int j = 0; j < num_groups; j++) {
            ASSERT_EQ(contact.get_cont_freq_mat().get_cont_freq(i, j),
                      read_contact.get_cont_freq_mat().get_cont_freq(i, j));

            ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size(),
                      read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size());
            for (int k = 0; k < contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector().size(); k++) {
                ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].day,
                          read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].day);
                ASSERT_EQ(contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].factor,
                          read_contact.get_cont_freq_mat().get_dampings(i, j).get_dampings_vector()[k].factor);
            }
        }
    }
}

TEST(TestSaveParameters, compareGraphs)
{
    double t0   = 0.0;
    double tmax = 50.5;
    double dt   = 0.1;

    double tinc = 5.2, tinfmild = 6, tserint = 4.2, thosp2home = 12, thome2hosp = 5, thosp2icu = 2, ticu2home = 8,
           tinfasy = 6.2, ticu2death = 5;

    double cont_freq = 0.5, alpha = 0.09, beta = 0.25, delta = 0.3, rho = 0.2, theta = 0.25;

    double num_total_t0 = 10000, num_exp_t0 = 100, num_inf_t0 = 50, num_car_t0 = 50, num_hosp_t0 = 20, num_icu_t0 = 10,
           num_rec_t0 = 10, num_dead_t0 = 0;

    int num_groups = 2;
    double fact    = 1.0 / (double)num_groups;

    epi::SecirParams params(num_groups);

    for (size_t i = 0; i < num_groups; i++) {
        params.times[i].set_incubation(tinc);
        params.times[i].set_infectious_mild(tinfmild);
        params.times[i].set_serialinterval(tserint);
        params.times[i].set_hospitalized_to_home(thosp2home);
        params.times[i].set_home_to_hospitalized(thome2hosp);
        params.times[i].set_hospitalized_to_icu(thosp2icu);
        params.times[i].set_icu_to_home(ticu2home);
        params.times[i].set_infectious_asymp(tinfasy);
        params.times[i].set_icu_to_death(ticu2death);

        params.populations.set({i, epi::SecirCompartments::E}, fact * num_exp_t0);
        params.populations.set({i, epi::SecirCompartments::C}, fact * num_car_t0);
        params.populations.set({i, epi::SecirCompartments::I}, fact * num_inf_t0);
        params.populations.set({i, epi::SecirCompartments::H}, fact * num_hosp_t0);
        params.populations.set({i, epi::SecirCompartments::U}, fact * num_icu_t0);
        params.populations.set({i, epi::SecirCompartments::R}, fact * num_rec_t0);
        params.populations.set({i, epi::SecirCompartments::D}, fact * num_dead_t0);
        params.populations.set_difference_from_group_total({i, epi::SecirCompartments::S}, epi::SecirCategory::AgeGroup,
                                                           i, fact * num_total_t0);

        params.probabilities[i].set_infection_from_contact(1.0);
        params.probabilities[i].set_asymp_per_infectious(alpha);
        params.probabilities[i].set_risk_from_symptomatic(beta);
        params.probabilities[i].set_hospitalized_per_infectious(rho);
        params.probabilities[i].set_icu_per_hospitalized(theta);
        params.probabilities[i].set_dead_per_icu(delta);
    }

    epi::ContactFrequencyMatrix& cont_freq_matrix = params.get_contact_patterns();
    epi::Damping dummy(30., 0.3);
    for (int i = 0; i < num_groups; i++) {
        for (int j = i; j < num_groups; j++) {
            cont_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
            cont_freq_matrix.add_damping(dummy, i, j);
        }
    }

    epi::ParameterSpace param_space(params, t0, tmax, 0.15);

    epi::Graph<epi::ModelNode<epi::SecirSimulation>, epi::MigrationEdge> graph;
    graph.add_node(param_space.get_secir_params(), t0);
    graph.add_node(param_space.get_secir_params(), t0);
    graph.add_edge(0, 1,
                   Eigen::VectorXd::Constant(param_space.get_secir_params().populations.get_num_compartments(), 0.01));
    graph.add_edge(1, 0,
                   Eigen::VectorXd::Constant(param_space.get_secir_params().populations.get_num_compartments(), 0.01));

    epi::write_graph(graph, t0, tmax);

    epi::Graph<epi::ModelNode<epi::SecirSimulation>, epi::MigrationEdge> graph_read = epi::read_graph();

    int num_nodes = graph.nodes().size();
    int num_edges = graph.edges().size();

    ASSERT_EQ(num_nodes, graph_read.nodes().size());
    ASSERT_EQ(num_edges, graph_read.edges().size());

    for (int node = 0; node < num_nodes; node++) {
        epi::SecirParams graph_params               = graph.nodes()[0].model.get_params();
        epi::ContactFrequencyMatrix graph_cont_freq = graph_params.get_contact_patterns();

        epi::SecirParams graph_read_params               = graph_read.nodes()[0].model.get_params();
        epi::ContactFrequencyMatrix graph_read_cont_freq = graph_read_params.get_contact_patterns();

        int num_compart = graph_params.populations.get_num_compartments() / num_groups;
        int num_groups  = graph_cont_freq.get_size();
        ASSERT_EQ(num_groups, graph_read_cont_freq.get_size());
        ASSERT_EQ(num_compart, graph_read_params.populations.get_num_compartments() / num_groups);

        int num_dampings = graph_cont_freq.get_dampings(0, 0).get_dampings_vector().size();
        ASSERT_EQ(num_dampings, graph_read_cont_freq.get_dampings(0, 0).get_dampings_vector().size());

        for (size_t group = 0; group < num_groups; group++) {
            ASSERT_EQ(graph_params.populations.get({group, epi::SecirCompartments::D}),
                      graph_read_params.populations.get({group, epi::SecirCompartments::D}));
            ASSERT_EQ(graph_params.populations.get_total(), graph_read_params.populations.get_total());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::E}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::E}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::C}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::C}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::I}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::I}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::H}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::H}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::U}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::U}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::R}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::R}).get_distribution().get());
            check_distribution(
                *graph_params.populations.get({group, epi::SecirCompartments::E}).get_distribution().get(),
                *graph_read_params.populations.get({group, epi::SecirCompartments::E}).get_distribution().get());

            // ASSERT_EQ(graph_params.times[group].get_incubation(), graph_read_params.times[group].get_incubation());
            check_distribution(*graph_params.times[group].get_incubation().get_distribution().get(),
                               *graph_read_params.times[group].get_incubation().get_distribution().get());
            check_distribution(*graph_params.times[group].get_serialinterval().get_distribution().get(),
                               *graph_read_params.times[group].get_serialinterval().get_distribution().get());
            check_distribution(*graph_params.times[group].get_infectious_mild().get_distribution().get(),
                               *graph_read_params.times[group].get_infectious_mild().get_distribution().get());
            check_distribution(*graph_params.times[group].get_hospitalized_to_home().get_distribution().get(),
                               *graph_read_params.times[group].get_hospitalized_to_home().get_distribution().get());
            check_distribution(*graph_params.times[group].get_home_to_hospitalized().get_distribution().get(),
                               *graph_read_params.times[group].get_home_to_hospitalized().get_distribution().get());
            check_distribution(*graph_params.times[group].get_infectious_asymp().get_distribution().get(),
                               *graph_read_params.times[group].get_infectious_asymp().get_distribution().get());
            check_distribution(*graph_params.times[group].get_hospitalized_to_icu().get_distribution().get(),
                               *graph_read_params.times[group].get_hospitalized_to_icu().get_distribution().get());
            check_distribution(*graph_params.times[group].get_icu_to_home().get_distribution().get(),
                               *graph_read_params.times[group].get_icu_to_home().get_distribution().get());
            check_distribution(*graph_params.times[group].get_icu_to_dead().get_distribution().get(),
                               *graph_read_params.times[group].get_icu_to_dead().get_distribution().get());

            check_distribution(
                *graph_params.probabilities[group].get_infection_from_contact().get_distribution().get(),
                *graph_read_params.probabilities[group].get_infection_from_contact().get_distribution().get());
            check_distribution(
                *graph_params.probabilities[group].get_asymp_per_infectious().get_distribution().get(),
                *graph_read_params.probabilities[group].get_asymp_per_infectious().get_distribution().get());
            check_distribution(
                *graph_params.probabilities[group].get_risk_from_symptomatic().get_distribution().get(),
                *graph_read_params.probabilities[group].get_risk_from_symptomatic().get_distribution().get());
            check_distribution(*graph_params.probabilities[group].get_dead_per_icu().get_distribution().get(),
                               *graph_read_params.probabilities[group].get_dead_per_icu().get_distribution().get());
            check_distribution(
                *graph_params.probabilities[group].get_hospitalized_per_infectious().get_distribution().get(),
                *graph_read_params.probabilities[group].get_hospitalized_per_infectious().get_distribution().get());
            check_distribution(
                *graph_params.probabilities[group].get_icu_per_hospitalized().get_distribution().get(),
                *graph_read_params.probabilities[group].get_icu_per_hospitalized().get_distribution().get());

            for (int contact_group = 0; contact_group < num_groups; contact_group++) {
                ASSERT_EQ(graph_cont_freq.get_cont_freq(group, contact_group),
                          graph_read_cont_freq.get_cont_freq(group, contact_group));
                for (int damp = 0; damp < num_dampings; damp++) {
                    ASSERT_EQ(
                        graph_cont_freq.get_dampings(group, contact_group).get_dampings_vector().at(damp).day,
                        graph_read_cont_freq.get_dampings(group, contact_group).get_dampings_vector().at(damp).day);
                    ASSERT_EQ(
                        graph_cont_freq.get_dampings(group, contact_group).get_dampings_vector().at(damp).factor,
                        graph_read_cont_freq.get_dampings(group, contact_group).get_dampings_vector().at(damp).factor);
                }
            }

            check_distribution(*graph_params.get_contact_patterns().get_distribution_damp_nb().get(),
                               *graph_read_params.get_contact_patterns().get_distribution_damp_nb().get());
            check_distribution(*graph_params.get_contact_patterns().get_distribution_damp_days().get(),
                               *graph_read_params.get_contact_patterns().get_distribution_damp_days().get());
            check_distribution(*graph_params.get_contact_patterns().get_distribution_damp_diag_base().get(),
                               *graph_read_params.get_contact_patterns().get_distribution_damp_diag_base().get());
            check_distribution(*graph_params.get_contact_patterns().get_distribution_damp_diag_rel().get(),
                               *graph_read_params.get_contact_patterns().get_distribution_damp_diag_rel().get());
            check_distribution(*graph_params.get_contact_patterns().get_distribution_damp_offdiag_rel().get(),
                               *graph_read_params.get_contact_patterns().get_distribution_damp_offdiag_rel().get());
        }

        for (int edge = 0; edge < num_edges; edge++) {
            ASSERT_EQ(graph.edges()[edge].start_node_idx, graph_read.edges()[edge].start_node_idx);
            ASSERT_EQ(graph.edges()[edge].end_node_idx, graph_read.edges()[edge].end_node_idx);
            for (int i = 0; i < num_compart * num_groups; i++) {
                ASSERT_EQ(graph.edges()[edge].property.coefficients[i],
                          graph_read.edges()[edge].property.coefficients[i]);
            }
        }
    }
}
