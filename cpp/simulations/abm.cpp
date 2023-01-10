/*
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
*
* Authors: Daniel Abele, Khoa Nguyen
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

#include "abm/abm.h"
#include "abm/household.h"
#include "abm/state.h"
#include "memilio/io/result_io.h"
#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

/**
 * Determine the infection state of a person at the beginning of the simulation.
 * The infection states are chosen randomly. They are distributed according to the probabilites set in the example.
 * @return random infection state
 */
mio::abm::InfectionState determine_infection_state(double exposed, double infected, double carrier, double recovered)
{
    double susceptible          = 1 - exposed - infected - carrier - recovered;
    std::vector<double> weights = {susceptible,  exposed,      carrier,       infected / 3,
                                   infected / 3, infected / 3, recovered / 2, recovered / 2};
    if (weights.size() != (size_t)mio::abm::InfectionState::Count - 1) {
        mio::log_error("Initialization in ABM wrong, please correct vector length.");
    }
    auto state = mio::DiscreteDistribution<size_t>::get_instance()(weights);
    return (mio::abm::InfectionState)state;
}


/**
 * Calculates a vector in which each entry describes the amount of people living in the corresponding household.
 * This is done with equal distribution and if the number of people is not divisible by number of households the last one gets the rest. E.g. number_of_people = 10, number_of_households = 3. Then the vector household_sizes = {3,3,4}.
 * @param number_of_people The total amount of people to be distributed.
 * @param number_of_households The total amount of households.
 * @return A vector with the size of each household.
 */
std::vector<int> last_household_gets_the_rest(int number_of_people, int number_of_households)
{
    std::vector<int> household_sizes(number_of_households, 0);
    int avarage_household_size_round_down = number_of_people / number_of_households; //int rounds down.
    int people_left                       = number_of_people -
                      avarage_household_size_round_down *
                          number_of_households; // People left if everyone got the same rounded down amount of people.
    for (auto i = 0; i < number_of_households - 1; i++) {
        household_sizes.at(i) = avarage_household_size_round_down;
    }
    household_sizes.at(number_of_households - 1) =
        avarage_household_size_round_down + people_left; // Last one gets the people which would've been left out.
    return household_sizes;
}

/**
 * Constructs a household group which has a single member to represent them all, e.g. all people have the same age distribution.
 * @param age_dist A vector with the amount of people in each age group
 * @param number_of_people The total amount of people living in this household group.
 * @param number_of_hh The number of households in this household group.
 * @return householdGroup A Class Household Group.
 */
mio::abm::HouseholdGroup make_uniform_households(const mio::abm::HouseholdMember& member, int number_of_people,
                                                 int number_of_hh)
{

    // The size of each household is calculated in a vector household_size_list.
    auto households_size_list = last_household_gets_the_rest(number_of_people, number_of_hh);

    auto householdGroup = mio::abm::HouseholdGroup();
    for (auto& household_size : households_size_list) {
        auto household = mio::abm::Household();
        household.add_members(member, household_size); // Add members according to the amount of people in the list.
        householdGroup.add_households(household, 1); // Add the household to the household group.

        // assuming 22 square meters per person and 3 meters of room height
        // see: https://doi.org/10.1371/journal.pone.0259037
        household.set_space_per_member(66);
    }
    return householdGroup;
}

/**
 * Constructs a household group with families.
 * @param child Child Household Member.
 * @param parent Parent Household Member.
 * @param random Random Household Member. This is for the rest Group where no exact age distribution can be found.
 * @param number_of_persons_in_household Amount of people in this household
 * @param number_of_full_familes Amount of full families, e.g. two parents and (number_of_persons_in_household - 2) children.
 * @param number_of_half_familes Amount of half families, e.g. one parent and (number_of_persons_in_household - 1) children.
 * @param number_of_other_familes number_of_persons_in_household random persons.
 * @return A Household group.
 */
mio::abm::HouseholdGroup make_homes_with_families(const mio::abm::HouseholdMember& child,
                                                  const mio::abm::HouseholdMember& parent,
                                                  const mio::abm::HouseholdMember& random,
                                                  int number_of_persons_in_household, int number_of_full_familes,
                                                  int number_of_half_familes, int number_of_other_familes)
{

    auto private_household_group = mio::abm::HouseholdGroup();

    // Add full families.
    auto household_full = mio::abm::Household();
    household_full.add_members(child, number_of_persons_in_household - 2);
    household_full.add_members(parent, 2);
    private_household_group.add_households(household_full, number_of_full_familes);

    // Add half families.
    auto household_half = mio::abm::Household();
    household_half.add_members(child, number_of_persons_in_household - 1);
    household_half.add_members(parent, 1);
    private_household_group.add_households(household_half, number_of_half_familes);

    // Add other families.
    if (number_of_persons_in_household < 5) {
        auto household_others = mio::abm::Household();
        household_others.add_members(random, number_of_persons_in_household);
        private_household_group.add_households(household_others, number_of_other_familes);
    }
    else if (number_of_persons_in_household == 5) {
        // For 5 and more people in one household we have to distribute the rest onto the left over households.
        int people_left_size5 = 545;

        auto households_size_list = last_household_gets_the_rest(people_left_size5, number_of_other_familes);

        auto household_rest = mio::abm::HouseholdGroup();
        for (auto& household_size : households_size_list) {
            auto household = mio::abm::Household();
            household.add_members(random, household_size); // Add members according to the amount of people in the list.
            household_rest.add_households(household, 1); // Add the household to the household group.
        }
    }
    return private_household_group;
}

void create_world_from_statistical_data(mio::abm::World& world)
{

    /** The data is taken from
     * https://www-genesis.destatis.de/genesis/online?operation=statistic&levelindex=0&levelid=1627908577036&code=12211#abreadcrumb
     * All numbers are in 1000.
     * Destatis divides the Households into community households and private households.
     * Community Households are: Refugee, Disabled, Retirement and Others. We have an explicit age distribution, amount of households and amount of people for them but not the exact amount of people in each household.
     * The private Households are divided with respect to the amount of people living in each household. For a one person household we have the exact age distribution. For the rest we have data about which kind of family lives in them. The different kinds of families are: A family with two parents and the rest are children, a family with one parent and the rest are children and  "other" families with no exact data about their age.
    */

    // Refugee
    auto refugee = mio::abm::HouseholdMember();
    refugee.set_age_weight(mio::abm::AgeGroup::Age0to4, 25);
    refugee.set_age_weight(mio::abm::AgeGroup::Age5to14, 12);
    refugee.set_age_weight(mio::abm::AgeGroup::Age15to34, 25);
    refugee.set_age_weight(mio::abm::AgeGroup::Age35to59, 9);
    refugee.set_age_weight(mio::abm::AgeGroup::Age60to79, 1);
    refugee.set_age_weight(mio::abm::AgeGroup::Age80plus, 1);
    int refugee_number_of_people     = 74;
    int refugee_number_of_households = 12;
    auto refugeeGroup = make_uniform_households(refugee, refugee_number_of_people, refugee_number_of_households);

    add_household_group_to_world(world, refugeeGroup);

    // Disabled
    auto disabled = mio::abm::HouseholdMember();
    disabled.set_age_weight(mio::abm::AgeGroup::Age0to4, 2);
    disabled.set_age_weight(mio::abm::AgeGroup::Age5to14, 6);
    disabled.set_age_weight(mio::abm::AgeGroup::Age15to34, 13);
    disabled.set_age_weight(mio::abm::AgeGroup::Age35to59, 42);
    disabled.set_age_weight(mio::abm::AgeGroup::Age60to79, 97);
    disabled.set_age_weight(mio::abm::AgeGroup::Age80plus, 32);
    int disabled_number_of_people     = 194;
    int disabled_number_of_households = 8;

    auto disabledGroup = make_uniform_households(disabled, disabled_number_of_people, disabled_number_of_households);

    add_household_group_to_world(world, disabledGroup);

    // Retirement
    auto retired = mio::abm::HouseholdMember();
    retired.set_age_weight(mio::abm::AgeGroup::Age15to34, 1);
    retired.set_age_weight(mio::abm::AgeGroup::Age35to59, 30);
    retired.set_age_weight(mio::abm::AgeGroup::Age60to79, 185);
    retired.set_age_weight(mio::abm::AgeGroup::Age80plus, 530);
    int retirement_number_of_people     = 744;
    int retirement_number_of_households = 16;

    auto retirementGroup =
        make_uniform_households(retired, retirement_number_of_people, retirement_number_of_households);

    add_household_group_to_world(world, retirementGroup);

    // Others
    auto other = mio::abm::HouseholdMember();
    other.set_age_weight(mio::abm::AgeGroup::Age0to4, 30);
    other.set_age_weight(mio::abm::AgeGroup::Age5to14, 40);
    other.set_age_weight(mio::abm::AgeGroup::Age15to34, 72);
    other.set_age_weight(mio::abm::AgeGroup::Age35to59, 40);
    other.set_age_weight(mio::abm::AgeGroup::Age60to79, 30);
    other.set_age_weight(mio::abm::AgeGroup::Age80plus, 10);
    int others_number_of_people     = 222;
    int others_number_of_households = 20;

    auto otherGroup = make_uniform_households(other, others_number_of_people, others_number_of_households);

    add_household_group_to_world(world, otherGroup);

    // One Person Household (we have exact age data about this)
    auto one_person_household_member = mio::abm::HouseholdMember();
    one_person_household_member.set_age_weight(mio::abm::AgeGroup::Age15to34, 4364);
    one_person_household_member.set_age_weight(mio::abm::AgeGroup::Age35to59, 7283);
    one_person_household_member.set_age_weight(mio::abm::AgeGroup::Age60to79, 4100);
    one_person_household_member.set_age_weight(mio::abm::AgeGroup::Age80plus, 1800);
    int one_person_number_of_people     = 15387;
    int one_person_number_of_households = 15387;

    auto onePersonGroup = make_uniform_households(one_person_household_member, one_person_number_of_people,
                                                  one_person_number_of_households);

    add_household_group_to_world(world, onePersonGroup);

    // For more than 1 family households we need families. These are parents and children and randoms (which are distributed like the data we have for these households).
    auto child = mio::abm::HouseholdMember(); // A child is 50/50% 0-4 or 5-14.
    child.set_age_weight(mio::abm::AgeGroup::Age0to4, 1);
    child.set_age_weight(mio::abm::AgeGroup::Age5to14, 1);

    auto parent = mio::abm::HouseholdMember(); // A child is 40/40/20% 15-34, 35-59 or 60-79.
    parent.set_age_weight(mio::abm::AgeGroup::Age15to34, 2);
    parent.set_age_weight(mio::abm::AgeGroup::Age35to59, 2);
    parent.set_age_weight(mio::abm::AgeGroup::Age60to79, 1);

    auto random = mio::abm::HouseholdMember(); // Randoms are distributed according to the left over persons.
    random.set_age_weight(mio::abm::AgeGroup::Age0to4, 5000);
    random.set_age_weight(mio::abm::AgeGroup::Age5to14, 6000);
    random.set_age_weight(mio::abm::AgeGroup::Age15to34, 14943);
    random.set_age_weight(mio::abm::AgeGroup::Age35to59, 22259);
    random.set_age_weight(mio::abm::AgeGroup::Age60to79, 11998);
    random.set_age_weight(mio::abm::AgeGroup::Age80plus, 5038);

    // Two person households
    int two_person_full_families  = 11850;
    int two_person_half_families  = 1765;
    int two_person_other_families = 166;
    auto twoPersonHouseholds      = make_homes_with_families(child, parent, random, 2, two_person_full_families,
                                                             two_person_half_families, two_person_other_families);
    add_household_group_to_world(world, twoPersonHouseholds);

    // Three person households
    int three_person_full_families  = 4155;
    int three_person_half_families  = 662;
    int three_person_other_families = 175;
    auto threePersonHouseholds      = make_homes_with_families(child, parent, random, 3, three_person_full_families,
                                                               three_person_half_families, three_person_other_families);
    add_household_group_to_world(world, threePersonHouseholds);

    // Four person households
    int four_person_full_families  = 3551;
    int four_person_half_families  = 110;
    int four_person_other_families = 122;
    auto fourPersonHouseholds      = make_homes_with_families(child, parent, random, 4, four_person_full_families,
                                                              four_person_half_families, four_person_other_families);
    add_household_group_to_world(world, fourPersonHouseholds);

    // Five plus person households
    int fiveplus_person_full_families  = 1245;
    int fiveplus_person_half_families  = 80;
    int fiveplus_person_other_families = 82;
    auto fivePlusPersonHouseholds =
        make_homes_with_families(child, parent, random, 5, fiveplus_person_full_families, fiveplus_person_half_families,
                                 fiveplus_person_other_families);
    add_household_group_to_world(world, fivePlusPersonHouseholds);
}

/**
 * Add locations to the world and assign locations to the people.
 */
void create_assign_locations(mio::abm::World& world)
{
    // Add one social event with 100 maximum contacts.
    // Maximum contacs limit the number of people that a person can infect while being at this location.
    // People have to get tested in the 2 days before the event
    // For the capacity we assume an area of 1.25 m^2 per person (https://doi.org/10.1371/journal.pone.0259037) and a
    // room height of 3 m
    auto event = world.add_location(mio::abm::LocationType::SocialEvent);
    world.get_individualized_location(event).get_infection_parameters().set<mio::abm::MaximumContacts>(100);
    world.get_individualized_location(event).set_capacity(100, 375);

    std::vector<mio::abm::LocationType> test_at_social_event = {mio::abm::LocationType::SocialEvent};
    auto testing_criteria =
        std::vector<mio::abm::TestingCriteria>{mio::abm::TestingCriteria({}, test_at_social_event, {})};
    auto testing_min_time = mio::abm::days(2);
    auto start_date       = mio::abm::TimePoint(0);
    auto end_date         = mio::abm::TimePoint(0) + mio::abm::days(60);
    auto probability      = 1.0;
    auto test_type        = mio::abm::AntigenTest();

    auto testing_scheme =
        mio::abm::TestingScheme(testing_criteria, testing_min_time, start_date, end_date, test_type, probability);

    world.get_testing_strategy().add_testing_scheme(testing_scheme);

    // Add hospital and ICU with 5 maximum contacs.
    // For the number of agents in this example we assume a capacity of 584 persons (80 beds per 10000 residents in
    // Germany (Statistisches Bundesamt, 2022) and a volume of 26242 m^3
    // (https://doi.org/10.1016/j.buildenv.2021.107926))
    // For the ICUs we assume a capacity of 30 agents and the same volume.
    auto hospital = world.add_location(mio::abm::LocationType::Hospital);
    world.get_individualized_location(hospital).get_infection_parameters().set<mio::abm::MaximumContacts>(5);
    world.get_individualized_location(hospital).set_capacity(584, 26242);
    auto icu = world.add_location(mio::abm::LocationType::ICU);
    world.get_individualized_location(icu).get_infection_parameters().set<mio::abm::MaximumContacts>(5);
    world.get_individualized_location(icu).set_capacity(30, 1350);

    // Add schools, workplaces and shops.
    // At every school there are 600 students. The maximum contacs are 40.
    // Students have to get tested once a week.
    // We assume 2 m^2 per student (https://doi.org/10.1371/journal.pone.0259037) and a room height of 3 m.
    // At every workplace work 100 people (needs to be varified), maximum contacts are 40.
    // People can get tested at work (and do this with 0.5 probability).
    // Per person we assume an area of 10 m^2 (https://doi.org/10.1371/journal.pone.0259037) and a room height of 3 m.
    // Add one supermarked per 15.000 people, maximum constacts are assumed to be 20.
    // A shop has a capacity of 240 persons (https://doi.org/10.1016/j.buildenv.2021.107926)
    // and a volume of 7200 cubic meters (10 m^2 per person (https://doi.org/10.1371/journal.pone.0259037) and 3 m
    // room height).
    auto shop = world.add_location(mio::abm::LocationType::BasicsShop);
    world.get_individualized_location(shop).get_infection_parameters().set<mio::abm::MaximumContacts>(20);
    world.get_individualized_location(shop).set_capacity(240, 7200);

    auto school = world.add_location(mio::abm::LocationType::School);
    world.get_individualized_location(school).get_infection_parameters().set<mio::abm::MaximumContacts>(40);
    world.get_individualized_location(school).set_capacity(600, 3600);

    auto work = world.add_location(mio::abm::LocationType::Work);
    world.get_individualized_location(work).get_infection_parameters().set<mio::abm::MaximumContacts>(40);
    world.get_individualized_location(work).set_capacity(100, 3000);

    int counter_event  = 0;
    int counter_school = 0;
    int counter_work   = 0;
    int counter_shop   = 0;
    //Assign locations to the people
    auto persons = world.get_persons();
    for (auto& person : persons) {
        //assign shop and event
        person.set_assigned_location(event);
        counter_event++;
        person.set_assigned_location(shop);
        counter_shop++;
        //assign hospital and ICU
        person.set_assigned_location(hospital);
        person.set_assigned_location(icu);
        //assign work/school to people depending on their age
        if (person.get_age() == mio::abm::AgeGroup::Age5to14) {
            person.set_assigned_location(school);
            counter_school++;
        }
        if (person.get_age() == mio::abm::AgeGroup::Age15to34 || person.get_age() == mio::abm::AgeGroup::Age35to59) {
            person.set_assigned_location(work);
            counter_work++;
        }
        //add new school/work/shop if needed
        if (counter_event == 1000) {
            counter_event = 0;
            event         = world.add_location(mio::abm::LocationType::SocialEvent);
            world.get_individualized_location(event).get_infection_parameters().set<mio::abm::MaximumContacts>(100);
            world.get_individualized_location(event).set_capacity(100, 375);
        }
        if (counter_school == 600) {
            counter_school = 0;
            school         = world.add_location(mio::abm::LocationType::School);
            world.get_individualized_location(school).get_infection_parameters().set<mio::abm::MaximumContacts>(40);
            world.get_individualized_location(school).set_capacity(600, 3600);
        }
        if (counter_work == 100) {
            counter_work = 0;
            work         = world.add_location(mio::abm::LocationType::Work);
            world.get_individualized_location(work).get_infection_parameters().set<mio::abm::MaximumContacts>(40);
            world.get_individualized_location(work).set_capacity(100, 3000);
        }
        if (counter_shop == 15000) {
            counter_shop = 0;
            shop         = world.add_location(mio::abm::LocationType::BasicsShop);
            world.get_individualized_location(shop).get_infection_parameters().set<mio::abm::MaximumContacts>(20);
            world.get_individualized_location(shop).set_capacity(240, 7200);
        }
    }

    // add the testing schemes for school and work
    auto test_at_school = std::vector<mio::abm::LocationType>{mio::abm::LocationType::School};
    auto testing_criteria_school =
        std::vector<mio::abm::TestingCriteria>{mio::abm::TestingCriteria({}, test_at_school, {})};

    testing_min_time           = mio::abm::days(7);
    probability                = 1.0;
    auto testing_scheme_school = mio::abm::TestingScheme(testing_criteria_school, testing_min_time, start_date,
                                                         end_date, test_type, probability);
    world.get_testing_strategy().add_testing_scheme(testing_scheme_school);

    auto test_at_work = std::vector<mio::abm::LocationType>{mio::abm::LocationType::Work};
    auto testing_criteria_work =
        std::vector<mio::abm::TestingCriteria>{mio::abm::TestingCriteria({}, test_at_work, {})};

    testing_min_time = mio::abm::days(1);
    probability      = 0.5;
    auto testing_scheme_work =
        mio::abm::TestingScheme(testing_criteria_work, testing_min_time, start_date, end_date, test_type, probability);
    world.get_testing_strategy().add_testing_scheme(testing_scheme_work);
}

/**
 * Assign an infection state to each person.
 */
void assign_infection_state(mio::abm::World& world, double exposed_pct, double infected_pct, double carrier_pct,
                            double recovered_pct)
{
    auto persons = world.get_persons();
    for (auto& person : persons) {
        world.set_infection_state(person,
                                  determine_infection_state(exposed_pct, infected_pct, carrier_pct, recovered_pct));
    }
}

void set_parameters(mio::abm::GlobalInfectionParameters infection_params)
{
    infection_params.set<mio::abm::IncubationPeriod>(
        {{mio::abm::AgeGroup::Count, mio::abm::VaccinationState::Count}, 4.});

    //0-4
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age0to4,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.05;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age0to4,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.05;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.276;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.092;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.142;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.001;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.186;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.015;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.143;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Unvaccinated}] =
        0.001;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age0to4,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    //5-14
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age5to14,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.1;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age5to14,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.1;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.276;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.092;
    infection_params.get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age5to14,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.142;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.001;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.186;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.015;
    infection_params.get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age5to14,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.143;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Unvaccinated}] =
        0.001;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age5to14,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    //15-34
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age15to34,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.13;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age15to34,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.13;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Unvaccinated}] =
        0.315;
    infection_params.get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age15to34,
                                                          mio::abm::VaccinationState::Unvaccinated}]  = 0.079;
    infection_params.get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age15to34,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.139;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Unvaccinated}] =
        0.003;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Unvaccinated}] =
        0.157;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Unvaccinated}] =
        0.013;
    infection_params.get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age15to34,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.126;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Unvaccinated}] =
        0.021;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age15to34,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    //35-59
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age35to59,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.11;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age35to59,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.11;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Unvaccinated}] =
        0.315;
    infection_params.get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age35to59,
                                                          mio::abm::VaccinationState::Unvaccinated}]  = 0.079;
    infection_params.get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age35to59,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.136;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Unvaccinated}] =
        0.009;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Unvaccinated}] =
        0.113;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Unvaccinated}] =
        0.02;
    infection_params.get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age35to59,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.05;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Unvaccinated}] =
        0.008;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age35to59,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    //60-79
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age60to79,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.04;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age60to79,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.04;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Unvaccinated}] =
        0.315;
    infection_params.get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age60to79,
                                                          mio::abm::VaccinationState::Unvaccinated}]  = 0.079;
    infection_params.get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age60to79,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.123;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Unvaccinated}] =
        0.024;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Unvaccinated}] =
        0.083;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Unvaccinated}] =
        0.035;
    infection_params.get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age60to79,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.035;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Unvaccinated}] =
        0.023;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age60to79,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    //80+
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age80plus,
                                                                     mio::abm::VaccinationState::Unvaccinated}]  = 0.07;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age80plus,
                                                                      mio::abm::VaccinationState::Unvaccinated}] = 0.07;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Unvaccinated}] =
        0.315;
    infection_params.get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age80plus,
                                                          mio::abm::VaccinationState::Unvaccinated}]  = 0.079;
    infection_params.get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age80plus,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.115;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Unvaccinated}] =
        0.033;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Unvaccinated}] =
        0.055;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Unvaccinated}] =
        0.036;
    infection_params.get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age80plus,
                                                           mio::abm::VaccinationState::Unvaccinated}] = 0.035;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Unvaccinated}] =
        0.052;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age80plus,
                                                              mio::abm::VaccinationState::Unvaccinated}] = 0.;

    // Set each parameter for vaccinated people

    //0-4
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age0to4,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.01;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age0to4,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.01;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.161;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.132;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.143;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.001;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.186;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.015;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] =
        0.143;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age0to4, mio::abm::VaccinationState::Vaccinated}] = 0.001;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age0to4,
                                                              mio::abm::VaccinationState::Vaccinated}]          = 0.0;

    //5-14
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age5to14,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.03;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age5to14,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.03;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.161;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.132;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.143;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.001;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.186;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.015;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.143;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age5to14, mio::abm::VaccinationState::Vaccinated}] =
        0.001;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age5to14,
                                                              mio::abm::VaccinationState::Vaccinated}] = 0.0;

    //15-34
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age15to34,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.03;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age15to34,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.03;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.179;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.126;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.142;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.001;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.157;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.013;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.126;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age15to34, mio::abm::VaccinationState::Vaccinated}] =
        0.021;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age15to34,
                                                              mio::abm::VaccinationState::Vaccinated}] = 0.0;

    //35-59
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age35to59,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.03;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age35to59,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.03;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.179;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.126;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.141;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.003;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.113;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.02;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.05;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age35to59, mio::abm::VaccinationState::Vaccinated}] =
        0.008;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age35to59,
                                                              mio::abm::VaccinationState::Vaccinated}] = 0.0;

    //60-79
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age60to79,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.01;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age60to79,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.01;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.179;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.126;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.136;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.009;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.083;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.035;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.035;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age60to79, mio::abm::VaccinationState::Vaccinated}] =
        0.023;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age60to79,
                                                              mio::abm::VaccinationState::Vaccinated}] = 0.0;

    //80+
    infection_params.get<mio::abm::SusceptibleToExposedByCarrier>()[{mio::abm::AgeGroup::Age80plus,
                                                                     mio::abm::VaccinationState::Vaccinated}]  = 0.02;
    infection_params.get<mio::abm::SusceptibleToExposedByInfected>()[{mio::abm::AgeGroup::Age80plus,
                                                                      mio::abm::VaccinationState::Vaccinated}] = 0.02;
    infection_params
        .get<mio::abm::CarrierToInfected>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.179;
    infection_params
        .get<mio::abm::CarrierToRecovered>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.126;
    infection_params
        .get<mio::abm::InfectedToRecovered>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.133;
    infection_params
        .get<mio::abm::InfectedToSevere>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.012;
    infection_params
        .get<mio::abm::SevereToRecovered>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.055;
    infection_params
        .get<mio::abm::SevereToCritical>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.036;
    infection_params
        .get<mio::abm::CriticalToRecovered>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.035;
    infection_params
        .get<mio::abm::CriticalToDead>()[{mio::abm::AgeGroup::Age80plus, mio::abm::VaccinationState::Vaccinated}] =
        0.052;
    infection_params.get<mio::abm::RecoveredToSusceptible>()[{mio::abm::AgeGroup::Age80plus,
                                                              mio::abm::VaccinationState::Vaccinated}] = 0.0;
}

/**
 * Create a sampled simulation with start time t0.
 * @param t0 the start time of the simulation
*/
mio::abm::Simulation create_sampled_simulation(const mio::abm::TimePoint& t0)
{

    // Assumed percentage of infection state at the beginning of the simulation.
    double exposed_pct = 0.005, infected_pct = 0.001, carrier_pct = 0.001, recovered_pct = 0.0;

    //Set global infection parameters (similar to infection parameters in SECIR model) and initialize the world
    mio::abm::GlobalInfectionParameters infection_params;
    set_parameters(infection_params);
    auto world = mio::abm::World(infection_params);

    // Create the world object from statistical data.
    create_world_from_statistical_data(world);

    // Assign an infection state to each person.
    assign_infection_state(world, exposed_pct, infected_pct, carrier_pct, recovered_pct);

    // Add locations and assign locations to the people.
    create_assign_locations(world);

    auto t_lockdown = mio::abm::TimePoint(0) + mio::abm::days(20);

    // During the lockdown, 25% of people work from home and schools are closed for 90% of students.
    // Social events are very rare.
    mio::abm::set_home_office(t_lockdown, 0.25, world.get_migration_parameters());
    mio::abm::set_school_closure(t_lockdown, 0.9, world.get_migration_parameters());
    mio::abm::close_social_events(t_lockdown, 0.9, world.get_migration_parameters());

    auto sim = mio::abm::Simulation(t0, std::move(world));
    return sim;
}

/**
 * Run the ABM simulation.
 * @param result_dir Directory where all results of the parameter study will be stored.
 * @param num_runs Number of runs.
 * @param save_single_runs [Default: true] Defines if single run results are written to the disk.
 * @returns Any io error that occurs during reading or writing of files.
 */
mio::IOResult<void> run(const fs::path& result_dir, size_t num_runs, bool save_single_runs = true)
{

    auto t0         = mio::abm::TimePoint(0); // Start time per simulation
    auto tmax       = mio::abm::TimePoint(0) + mio::abm::days(60); // End time per simulation
    auto ensemble_results = std::vector<std::vector<mio::TimeSeries<double>>>{}; // Vector of collected results
    ensemble_results.reserve(size_t(num_runs));
    auto run_idx            = size_t(1); // The run index
    auto save_result_result = mio::IOResult<void>(mio::success()); // Variable informing over successful IO operations

    // Loop over a number of runs
    while (run_idx <= num_runs) {
        
        // Create the sampled simulation with start time t0.
        auto sim = create_sampled_simulation(t0);
        // Collect the id of location in world.
        std::vector<int> loc_ids;
        for (auto&& locations : sim.get_world().get_locations()) {
            for (auto location : locations){
                loc_ids.push_back(location.get_index());
            }
        }
        // Advance the world to tmax
        sim.advance(tmax);
        // TODO: update result of the simulation to be a vector of location result.
        auto temp_sim_result = std::vector<mio::TimeSeries<double>>{sim.get_result()};
        // Push result of the simulation back to the result vector
        ensemble_results.push_back(temp_sim_result);
        // Option to save the current run result to file
        if (save_result_result && save_single_runs) {
            auto result_dir_run = result_dir / ("abm_result_run_" + std::to_string(run_idx) + ".h5");
            BOOST_OUTCOME_TRY(save_result(ensemble_results.back(), loc_ids, 1, result_dir_run.string()));
        }
        ++run_idx;
    }
    BOOST_OUTCOME_TRY(save_result_result);
    return mio::success();
}

int main(int argc, char** argv)
{

    mio::set_log_level(mio::LogLevel::warn);

    std::string result_dir = ".";
    size_t num_runs;
    bool save_single_runs = true;

    if (argc == 2) {
        num_runs = atoi(argv[1]);
        printf("Number of run is %s.\n", argv[1]);
        printf("Saving results to the current directory.\n");
    } else if (argc == 3) {
        num_runs = atoi(argv[1]);
        result_dir = argv[2];
        printf("Number of run is %s.\n", argv[1]);
        printf("Saving results to \"%s\".\n", result_dir.c_str());
    } else {
        printf("Usage:\n");
        printf("abm_example <num_runs>\n");
        printf("\tRun the simulation for <num_runs> time(s).\n");
        printf("\tStore the results in the current directory.\n");
        printf("abm_example <num_runs> <result_dir>\n");
        printf("\tRun the simulation for <num_runs> time(s).\n");
        printf("\tStore the results in <result_dir>.\n");
        return 0;
    }

    // mio::thread_local_rng().seed({...}); //set seeds, e.g., for debugging
    //printf("Seeds: ");
    //for (auto s : mio::thread_local_rng().get_seeds()) {
    //    printf("%u, ", s);
    //}
    //printf("\n");

    auto result = run(result_dir, num_runs, save_single_runs);
    if (!result) {
        printf("%s\n", result.error().formatted_message().c_str());
        return -1;
    }
    return 0;
}