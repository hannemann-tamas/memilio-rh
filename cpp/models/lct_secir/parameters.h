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

#ifndef LCT_SECIR_PARAMS_H
#define LCT_SECIR_PARAMS_H

#include "memilio/config.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/math/eigen.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/utils/logging.h"

namespace mio
{
namespace lsecir
{

/**********************************************
* Define Parameters of the LCT-SECIHURD model *
**********************************************/

/**
 * @brief Average Time spent in the Exposed compartment.
 */
struct TimeExposed {
    using Type = ScalarType;
    static Type get_default()
    {
        return 2.0;
    }
    static std::string name()
    {
        return "TimeExposed";
    }
};

/**
 * @brief Average time spent in the TimeInfectedNoSymptoms before developing 
 *  Symptoms or recover in the SECIR model in day unit.
 */
struct TimeInfectedNoSymptoms {
    using Type = ScalarType;
    static Type get_default()
    {
        return 1.0;
    }
    static std::string name()
    {
        return "TimeInfectedNoSymptoms";
    }
};

/**
 * @brief Average time spent in the TimeInfectedSymptoms before going to Hospital 
 *  or recover in the SECIR model in day unit.
 */
struct TimeInfectedSymptoms {
    using Type = ScalarType;
    static Type get_default()
    {
        return 1.5;
    }
    static std::string name()
    {
        return "TimeInfectedNoSymptoms";
    }
};

/**
 * @brief Average time being in the Hospital before treated by ICU or recover in the 
 *  SECIR model in day unit.
 */
struct TimeInfectedSevere {
    using Type = ScalarType;
    static Type get_default()
    {
        return 1.0;
    }
    static std::string name()
    {
        return "TimeInfectedSevere";
    }
};

/**
 * @brief Average time treated by ICU before dead or recover in the SECIR model in day unit.
 */
struct TimeInfectedCritical {
    using Type = ScalarType;
    static Type get_default()
    {
        return 1.0;
    }
    static std::string name()
    {
        return "TimeInfectedCritical";
    }
};

/**
 * @brief Probability of getting infected from a contact.
 */
struct TransmissionProbabilityOnContact {
    using Type = ScalarType;
    static Type get_default()
    {
        return 1.0;
    }
    static std::string name()
    {
        return "TransmissionProbabilityOnContact";
    }
};

/**
 * @brief The contact patterns within the society are modelled using an UncertainContactMatrix.
 */
struct ContactPatterns {
    using Type = UncertainContactMatrix<ScalarType>;

    static Type get_default()
    {
        ContactMatrixGroup contact_matrix = ContactMatrixGroup(1, 1);
        contact_matrix[0]                 = mio::ContactMatrix(Eigen::MatrixXd::Constant(1, 1, 10.));
        return Type(contact_matrix);
    }
    static std::string name()
    {
        return "ContactPatterns";
    }
};

/**
 * @brief The relative InfectedNoSymptoms infectability.
 */
struct RelativeTransmissionNoSymptoms {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.5;
    }
    static std::string name()
    {
        return "RelativeTransmissionNoSymptoms";
    }
};

/**
 * @brief The risk of infection from symptomatic cases in the SECIR model.
 */
struct RiskOfInfectionFromSymptomatic {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.5;
    }
    static std::string name()
    {
        return "RiskOfInfectionFromSymptomatic";
    }
};

/**
 * @brief The percentage of asymptomatic cases in the SECIR model.
 */
struct RecoveredPerInfectedNoSymptoms {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.5;
    }
    static std::string name()
    {
        return "RecoveredPerInfectedNoSymptoms";
    }
};

/**
 * @brief The percentage of hospitalized patients per infected patients in the SECIR model.
 */
struct SeverePerInfectedSymptoms {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.5;
    }
    static std::string name()
    {
        return "SeverePerInfectedSymptoms";
    }
};

/**
 * @brief The percentage of ICU patients per hospitalized patients in the SECIR model.
 */
struct CriticalPerSevere {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.5;
    }
    static std::string name()
    {
        return "CriticalPerSevere";
    }
};

/**
 * @brief The percentage of dead patients per ICU patients in the SECIR model.
 */
struct DeathsPerCritical {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.1;
    }
    static std::string name()
    {
        return "DeathsPerCritical";
    }
};

/**
 * @brief The start day in the LCT SECIR model.
 * The start day defines in which season the simulation is started.
 * If the start day is 180 and simulation takes place from t0=0 to
 * tmax=100 the days 180 to 280 of the year are simulated.
 */
struct StartDay {
    using Type = ScalarType;
    static Type get_default()
    {
        return 0.;
    }
    static std::string name()
    {
        return "StartDay";
    }
};

/**
 * @brief The seasonality in the LCT-SECIR model.
 * The seasonality is given as (1+k*sin()) where the sine
 * curve is below one in summer and above one in winter.
 */
struct Seasonality {
    using Type = ScalarType;
    static Type get_default()
    {
        return Type(0.);
    }
    static std::string name()
    {
        return "Seasonality";
    }
};

using ParametersBase =
    ParameterSet<TimeExposed, TimeInfectedNoSymptoms, TimeInfectedSymptoms, TimeInfectedSevere, TimeInfectedCritical,
                 TransmissionProbabilityOnContact, ContactPatterns, RelativeTransmissionNoSymptoms,
                 RiskOfInfectionFromSymptomatic, RecoveredPerInfectedNoSymptoms, SeverePerInfectedSymptoms,
                 CriticalPerSevere, DeathsPerCritical, StartDay, Seasonality>;

/**
 * @brief Parameters of an LCT-SECIR model.
 */
class Parameters : public ParametersBase
{
public:
    /**
     * @brief Default constructor.
     */
    Parameters()
        : ParametersBase()
    {
    }

    /**
     * @brief checks whether all Parameters satisfy their corresponding constraints and throws errors, if they do not.
     * @return Returns true if one (or more) constraint(s) are not satisfied, otherwise false. 
     */
    bool check_constraints() const
    {
        if (this->get<TimeExposed>() < 1.0) {
            log_error("Constraint check: Parameter TimeExposed is smaller than {:.4f}", 1.0);
            return true;
        }

        if (this->get<TimeInfectedNoSymptoms>() < 1.0) {
            log_error("Constraint check: Parameter TimeInfectedNoSymptoms is smaller than {:.4f}", 1.0);
            return true;
        }

        if (this->get<TimeInfectedSymptoms>() < 1.0) {
            log_error("Constraint check: Parameter TimeInfectedSymptoms is smaller than {:.4f}", 1.0);
            return true;
        }

        if (this->get<TimeInfectedSevere>() < 1.0) {
            log_error("Constraint check: Parameter TimeInfectedSevere is smaller than {:.4f}", 1.0);
            return true;
        }

        if (this->get<TimeInfectedCritical>() < 1.0) {
            log_error("Constraint check: Parameter TimeInfectedCritical is smaller than {:.4f}", 1.0);
            return true;
        }

        if (this->get<TransmissionProbabilityOnContact>() < 0.0 ||
            this->get<TransmissionProbabilityOnContact>() > 1.0) {
            log_error("Constraint check: Parameter TransmissionProbabilityOnContact smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<RelativeTransmissionNoSymptoms>() < 0.0 || this->get<RelativeTransmissionNoSymptoms>() > 1.0) {
            log_error("Constraint check: Parameter RelativeTransmissionNoSymptoms smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<RiskOfInfectionFromSymptomatic>() < 0.0 || this->get<RiskOfInfectionFromSymptomatic>() > 1.0) {
            log_error("Constraint check: Parameter  RiskOfInfectionFromSymptomatic smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<RecoveredPerInfectedNoSymptoms>() < 0.0 || this->get<RecoveredPerInfectedNoSymptoms>() > 1.0) {
            log_error("Constraint check: Parameter RecoveredPerInfectedNoSymptoms smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<SeverePerInfectedSymptoms>() < 0.0 || this->get<SeverePerInfectedSymptoms>() > 1.0) {
            log_error("Constraint check: Parameter SeverePerInfectedSymptoms smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<CriticalPerSevere>() < 0.0 || this->get<CriticalPerSevere>() > 1.0) {
            log_error("Constraint check: Parameter CriticalPerSevere smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<DeathsPerCritical>() < 0.0 || this->get<DeathsPerCritical>() > 1.0) {
            log_error("Constraint check: Parameter DeathsPerCritical smaller {:d} or larger {:d}", 0, 1);
            return true;
        }

        if (this->get<Seasonality>() < 0.0 || this->get<Seasonality>() > 0.5) {
            log_warning("Constraint check: Parameter Seasonality should lie between {:0.4f} and {:.4f}", 0.0, 0.5);
            return true;
        }

        return false;
    }

private:
    Parameters(ParametersBase&& base)
        : ParametersBase(std::move(base))
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
        BOOST_OUTCOME_TRY(auto&& base, ParametersBase::deserialize(io));
        return success(Parameters(std::move(base)));
    }
};

} // namespace lsecir
} // namespace mio

#endif // LCT_SECIR_PARAMS_H
