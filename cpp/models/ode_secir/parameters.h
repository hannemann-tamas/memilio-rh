/* 
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
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
#ifndef SECIR_PARAMETERS_H
#define SECIR_PARAMETERS_H

#include "memilio/math/eigen.h"
#include "memilio/utils/uncertain_value.h"
#include "memilio/math/adapt_rk.h"
#include "memilio/epidemiology/age_group.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/epidemiology/dynamic_npis.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/custom_index_array.h"

#include <vector>

namespace mio
{
namespace osecir
{

/*******************************************
 * Define Parameters of the SECIHURD model *
 *******************************************/

/**
 * @brief the start day in the SECIR model
 * The start day defines in which season the simulation can be started
 * If the start day is 180 and simulation takes place from t0=0 to
 * tmax=100 the days 180 to 280 of the year are simulated
 */
struct StartDay {
    using Type = double;
    static Type get_default(AgeGroup)
    {
        return 0.;
    }
    static std::string name()
    {
        return "StartDay";
    }
};

/**
 * @brief the seasonality in the SECIR model
 * the seasonality is given as (1+k*sin()) where the sine
 * curve is below one in summer and above one in winter
 */
template<typename FP=double>
struct Seasonality {
    using Type = UncertainValue<FP>;
    static Type get_default(AgeGroup)
    {
        return Type(0.);
    }
    static std::string name()
    {
        return "Seasonality";
    }
};

/**
 * @brief the icu capacity in the SECIR model
 */
template<typename FP=double>
struct ICUCapacity {
    using Type = UncertainValue<FP>;
    static Type get_default(AgeGroup)
    {
        return Type(std::numeric_limits<FP>::max());
    }
    static std::string name()
    {
        return "ICUCapacity";
    }
};

/**
 * @brief the incubation time in the SECIR model
 * @param tinc incubation time in day unit
 */
template<typename FP=double>
struct IncubationTime {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 2.);
    }
    static std::string name()
    {
        return "IncubationTime";
    }
};

/**
 * @brief the infectious time for symptomatic cases that are infected but
 *        who do not need to be hsopitalized in the SECIR model in day unit
 */
template<typename FP=double>
struct TimeInfectedSymptoms {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.);
    }
    static std::string name()
    {
        return "TimeInfectedSymptoms";
    }
};

/**
 * @brief the serial interval in the SECIR model in day unit
 */
template<typename FP=double>
struct SerialInterval {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.5);
    }
    static std::string name()
    {
        return "SerialInterval";
    }
};

/**
 * @brief the time people are 'simply' hospitalized before returning home in the SECIR model
 *        in day unit
 */
template<typename FP=double>
struct TimeInfectedSevere {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.);
    }
    static std::string name()
    {
        return "TimeInfectedSevere";
    }
};

/**
 * @brief the time people are treated by ICU before returning home in the SECIR model
 *        in day unit
 */
template<typename FP=double>
struct TimeInfectedCritical {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.);
    }
    static std::string name()
    {
        return "TimeInfectedCritical";
    }
};

/**
* @brief probability of getting infected from a contact
*/
template<typename FP=double>
struct TransmissionProbabilityOnContact {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.);
    }
    static std::string name()
    {
        return "TransmissionProbabilityOnContact";
    }
};

/**
* @brief the relative InfectedNoSymptoms infectability
*/
template<typename FP=double>
struct RelativeTransmissionNoSymptoms {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 1.);
    }
    static std::string name()
    {
        return "RelativeTransmissionNoSymptoms";
    }
};

/**
* @brief the percentage of asymptomatic cases in the SECIR model
*/
template<typename FP=double>
struct RecoveredPerInfectedNoSymptoms {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "RecoveredPerInfectedNoSymptoms";
    }
};

/**
* @brief the risk of infection from symptomatic cases in the SECIR model
*/
template<typename FP=double>
struct RiskOfInfectionFromSymptomatic {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "RiskOfInfectionFromSymptomatic";
    }
};

/**
* @brief risk of infection from symptomatic cases increases as test and trace capacity is exceeded.
*/
template<typename FP=double>
struct MaxRiskOfInfectionFromSymptomatic {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "MaxRiskOfInfectionFromSymptomatic";
    }
};

/**
* @brief the percentage of hospitalized patients per infected patients in the SECIR model
*/
template<typename FP=double>
struct SeverePerInfectedSymptoms {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "SeverePerInfectedSymptoms";
    }
};

/**
* @brief the percentage of ICU patients per hospitalized patients in the SECIR model
*/
template<typename FP=double>
struct CriticalPerSevere {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "CriticalPerSevere";
    }
};

/**
* @brief the percentage of dead patients per ICU patients in the SECIR model
*/
template<typename FP=double>
struct DeathsPerCritical {
    using Type = CustomIndexArray<UncertainValue<FP>, AgeGroup>;
    static Type get_default(AgeGroup size)
    {
        return Type(size, 0.);
    }
    static std::string name()
    {
        return "DeathsPerCritical";
    }
};

/**
 * @brief the contact patterns within the society are modelled using an UncertainContactMatrix
 */
template<typename FP=double>
struct ContactPatterns {
    using Type = UncertainContactMatrix<FP>;
    static Type get_default(AgeGroup size)
    {
        return Type(1, static_cast<Eigen::Index>((size_t)size));
    }
    static std::string name()
    {
        return "ContactPatterns";
    }
};

/**
 * @brief the NPIs that are enacted if certain infection thresholds are exceeded.
 */
template<typename FP=double>
struct DynamicNPIsInfectedSymptoms {
    using Type = DynamicNPIs<FP>;
    static Type get_default(AgeGroup /*size*/)
    {
        return {};
    }
    static std::string name()
    {
        return "DynamicNPIsInfectedSymptoms";
    }
};

/**
 * @brief capacity to test and trace contacts of infected for quarantine per day.
 */
template<typename FP=double>
struct TestAndTraceCapacity {
    using Type = UncertainValue<FP>;
    static Type get_default(AgeGroup)
    {
        return Type(std::numeric_limits<FP>::max());
    }
    static std::string name()
    {
        return "TestAndTraceCapacity";
    }
};

template<typename FP=double>
using ParametersBase =
    ParameterSet<StartDay, Seasonality<FP>, ICUCapacity<FP>, TestAndTraceCapacity<FP>, ContactPatterns<FP>, DynamicNPIsInfectedSymptoms<FP>,
                 IncubationTime<FP>, TimeInfectedSymptoms<FP>, SerialInterval<FP>, TimeInfectedSevere<FP>, TimeInfectedCritical<FP>,
                 TransmissionProbabilityOnContact<FP>, RelativeTransmissionNoSymptoms<FP>, RecoveredPerInfectedNoSymptoms<FP>,
                 RiskOfInfectionFromSymptomatic<FP>, MaxRiskOfInfectionFromSymptomatic<FP>, SeverePerInfectedSymptoms<FP>,
                 CriticalPerSevere<FP>, DeathsPerCritical<FP>>;

/**
 * @brief Parameters of an age-resolved SECIR/SECIHURD model.
 */
template<typename FP=double>
class Parameters : public ParametersBase<FP>
{
public:
    Parameters(AgeGroup num_agegroups)
        : ParametersBase<FP>(num_agegroups)
        , m_num_groups{num_agegroups}
    {
    }

    AgeGroup get_num_groups() const
    {
        return m_num_groups;
    }

    /**
     * @brief checks whether all Parameters satisfy their corresponding constraints and applies them, if they do not
     */
    void apply_constraints()
    {
        if (this->template get<Seasonality<FP>>() < 0.0 || this->template get<Seasonality<FP>>() > 0.5) {
            log_warning("Constraint check: Parameter Seasonality changed from {:0.4f} to {:d}",
                        this->template get<Seasonality<FP>>(), 0);
            this->template set<Seasonality<FP>>(0);
        }

        if (this->template get<ICUCapacity<FP>>() < 0.0) {
            log_warning("Constraint check: Parameter ICUCapacity changed from {:0.4f} to {:d}",
                        this->template get<ICUCapacity<FP>>(), 0);
            this->template set<ICUCapacity<FP>>(0);
        }

        for (auto i = AgeGroup(0); i < AgeGroup(m_num_groups); ++i) {

            if (this->template get<IncubationTime<FP>>()[i] < 2.0) {
                log_warning("Constraint check: Parameter IncubationTime changed from {:.4f} to {:.4f}",
                            this->template get<IncubationTime<FP>>()[i], 2.0);
                this->template get<IncubationTime<FP>>()[i] = 2.0;
            }

            if (2 * this->template get<SerialInterval<FP>>()[i] < this->template get<IncubationTime<FP>>()[i] + 1.0) {
                log_warning("Constraint check: Parameter SerialInterval changed from {:.4f} to {:.4f}",
                            this->template get<SerialInterval<FP>>()[i], 0.5 * this->template get<IncubationTime<FP>>()[i] + 0.5);
                this->template get<SerialInterval<FP>>()[i] = 0.5 * this->template get<IncubationTime<FP>>()[i] + 0.5;
            }
            else if (this->template get<SerialInterval<FP>>()[i] > this->template get<IncubationTime<FP>>()[i] - 0.5) {
                log_warning("Constraint check: Parameter SerialInterval changed from {:.4f} to {:.4f}",
                            this->template get<SerialInterval<FP>>()[i], this->template get<IncubationTime<FP>>()[i] - 0.5);
                this->template get<SerialInterval<FP>>()[i] = this->template get<IncubationTime<FP>>()[i] - 0.5;
            }

            if (this->template get<TimeInfectedSymptoms<FP>>()[i] < 1.0) {
                log_warning("Constraint check: Parameter TimeInfectedSymptoms changed from {:.4f} to {:.4f}",
                            this->template get<TimeInfectedSymptoms<FP>>()[i], 1.0);
                this->template get<TimeInfectedSymptoms<FP>>()[i] = 1.0;
            }

            if (this->template get<TimeInfectedSevere<FP>>()[i] < 1.0) {
                log_warning("Constraint check: Parameter TimeInfectedSevere changed from {:.4f} to {:.4f}",
                            this->template get<TimeInfectedSevere<FP>>()[i], 1.0);
                this->template get<TimeInfectedSevere<FP>>()[i] = 1.0;
            }

            if (this->template get<TimeInfectedCritical<FP>>()[i] < 1.0) {
                log_warning("Constraint check: Parameter TimeInfectedCritical changed from {:.4f} to {:.4f}",
                            this->template get<TimeInfectedCritical<FP>>()[i], 1.0);
                this->template get<TimeInfectedCritical<FP>>()[i] = 1.0;
            }

            if (this->template get<TransmissionProbabilityOnContact<FP>>()[i] < 0.0) {
                log_warning(
                    "Constraint check: Parameter TransmissionProbabilityOnContact changed from {:0.4f} to {:d} ",
                    this->template get<TransmissionProbabilityOnContact<FP>>()[i], 0);
                this->template get<TransmissionProbabilityOnContact<FP>>()[i] = 0;
            }

            if (this->template get<RelativeTransmissionNoSymptoms<FP>>()[i] < 0.0) {
                log_warning("Constraint check: Parameter RelativeTransmissionNoSymptoms changed from {:0.4f} to {:d} ",
                            this->template get<RelativeTransmissionNoSymptoms<FP>>()[i], 0);
                this->template get<RelativeTransmissionNoSymptoms<FP>>()[i] = 0;
            }

            if (this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i] < 0.0 ||
                this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i] > 1.0) {
                log_warning("Constraint check: Parameter RecoveredPerInfectedNoSymptoms changed from {:0.4f} to {:d} ",
                            this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i], 0);
                this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i] = 0;
            }

            if (this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i] < 0.0 ||
                this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i] > 1.0) {
                log_warning("Constraint check: Parameter RiskOfInfectionFromSymptomatic changed from {:0.4f} to {:d}",
                            this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i], 0);
                this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i] = 0;
            }

            if (this->template get<SeverePerInfectedSymptoms<FP>>()[i] < 0.0 || this->template get<SeverePerInfectedSymptoms<FP>>()[i] > 1.0) {
                log_warning("Constraint check: Parameter SeverePerInfectedSymptoms changed from {:0.4f} to {:d}",
                            this->template get<SeverePerInfectedSymptoms<FP>>()[i], 0);
                this->template get<SeverePerInfectedSymptoms<FP>>()[i] = 0;
            }

            if (this->template get<CriticalPerSevere<FP>>()[i] < 0.0 || this->template get<CriticalPerSevere<FP>>()[i] > 1.0) {
                log_warning("Constraint check: Parameter CriticalPerSevere changed from {:0.4f} to {:d}",
                            this->template get<CriticalPerSevere<FP>>()[i], 0);
                this->template get<CriticalPerSevere<FP>>()[i] = 0;
            }

            if (this->template get<DeathsPerCritical<FP>>()[i] < 0.0 || this->template get<DeathsPerCritical<FP>>()[i] > 1.0) {
                log_warning("Constraint check: Parameter DeathsPerCritical changed from {:0.4f} to {:d}",
                            this->template get<DeathsPerCritical<FP>>()[i], 0);
                this->template get<DeathsPerCritical<FP>>()[i] = 0;
            }
        }
    }

    /**
     * @brief Checks whether all Parameters satisfy their corresponding constraints and logs an error 
     * if constraints are not satisfied.
     * @return Returns 1 if one constraint is not satisfied, otherwise 0.   
     */
    int check_constraints() const
    {
        if (this->template get<Seasonality<FP>>() < 0.0 || this->template get<Seasonality<FP>>() > 0.5) {
            log_error("Constraint check: Parameter Seasonality smaller {:d} or larger {:d}", 0, 0.5);
            return 1;
        }

        if (this->template get<ICUCapacity<FP>>() < 0.0) {
            log_error("Constraint check: Parameter ICUCapacity smaller {:d}", 0);
            return 1;
        }

        for (auto i = AgeGroup(0); i < AgeGroup(m_num_groups); ++i) {

            if (this->template get<IncubationTime<FP>>()[i] < 2.0) {
                log_error("Constraint check: Parameter IncubationTime {:.4f} smaller {:.4f}",
                          this->template get<IncubationTime<FP>>()[i], 2.0);
                return 1;
            }

            if (2 * this->template get<SerialInterval<FP>>()[i] < this->template get<IncubationTime<FP>>()[i] + 1.0) {
                log_error("Constraint check: Parameter SerialInterval {:.4f} smaller {:.4f}",
                          this->template get<SerialInterval<FP>>()[i], 0.5 * this->template get<IncubationTime<FP>>()[i] + 0.5);
                return 1;
            }
            else if (this->template get<SerialInterval<FP>>()[i] > this->template get<IncubationTime<FP>>()[i] - 0.5) {
                log_error("Constraint check: Parameter SerialInterval {:.4f} greater {:.4f}",
                          this->template get<SerialInterval<FP>>()[i], this->template get<IncubationTime<FP>>()[i] - 0.5);
                return 1;
            }

            if (this->template get<TimeInfectedSymptoms<FP>>()[i] < 1.0) {
                log_error("Constraint check: Parameter TimeInfectedSymptoms {:.4f} smaller {:.4f}",
                          this->template get<TimeInfectedSymptoms<FP>>()[i], 1.0);
                return 1;
            }

            if (this->template get<TimeInfectedSevere<FP>>()[i] < 1.0) {
                log_error("Constraint check: Parameter TimeInfectedSevere {:.4f} smaller {:.4f}",
                          this->template get<TimeInfectedSevere<FP>>()[i], 1.0);
                return 1;
            }

            if (this->template get<TimeInfectedCritical<FP>>()[i] < 1.0) {
                log_error("Constraint check: Parameter TimeInfectedCritical {:.4f} smaller {:.4f}",
                          this->template get<TimeInfectedCritical<FP>>()[i], 1.0);
                return 1;
            }

            if (this->template get<TransmissionProbabilityOnContact<FP>>()[i] < 0.0 ||
                this->template get<TransmissionProbabilityOnContact<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter TransmissionProbabilityOnContact smaller {:d} or larger {:d}", 0,
                          1);
                return 1;
            }

            if (this->template get<RelativeTransmissionNoSymptoms<FP>>()[i] < 0.0) {
                log_error("Constraint check: Parameter RelativeTransmissionNoSymptoms smaller {:d}", 0);
                return 1;
            }

            if (this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i] < 0.0 ||
                this->template get<RecoveredPerInfectedNoSymptoms<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter RecoveredPerInfectedNoSymptoms smaller {:d} or larger {:d}", 0,
                          1);
                return 1;
            }

            if (this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i] < 0.0 ||
                this->template get<RiskOfInfectionFromSymptomatic<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter RiskOfInfectionFromSymptomatic smaller {:d} or larger {:d}", 0,
                          1);
                return 1;
            }

            if (this->template get<SeverePerInfectedSymptoms<FP>>()[i] < 0.0 || this->template get<SeverePerInfectedSymptoms<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter SeverePerInfectedSymptoms smaller {:d} or larger {:d}", 0, 1);
                return 1;
            }

            if (this->template get<CriticalPerSevere<FP>>()[i] < 0.0 || this->template get<CriticalPerSevere<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter CriticalPerSevere smaller {:d} or larger {:d}", 0, 1);
                return 1;
            }

            if (this->template get<DeathsPerCritical<FP>>()[i] < 0.0 || this->template get<DeathsPerCritical<FP>>()[i] > 1.0) {
                log_error("Constraint check: Parameter DeathsPerCritical smaller {:d} or larger {:d}", 0, 1);
                return 1;
            }
        }
        return 0;
    }

private:
    Parameters(ParametersBase<FP>&& base)
        : ParametersBase<FP>(std::move(base))
        , m_num_groups(this->template get<ContactPatterns<FP>>().get_cont_freq_mat().get_num_groups())
    {
    }

public:
    /**
     * deserialize an object of this class.
     * @see mio::deserialize
     */
    template <class IOContext>
    static IOResult<Parameters> deserialize(IOContext& io)
    {
        BOOST_OUTCOME_TRY(base, ParametersBase<FP>::deserialize(io));
        return success(Parameters(std::move(base)));
    }

private:
    AgeGroup m_num_groups;
};

/**
 * @brief WIP !! TO DO: returns the actual, approximated reproduction rate 
 */
//double get_reprod_rate(Parameters const& params, double t, std::vector<double> const& yt);

} // namespace osecir
} // namespace mio

#endif // SECIR_PARAMETERS_H
