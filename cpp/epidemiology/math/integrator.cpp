#include "epidemiology/math/integrator.h"
#include <epidemiology/utils/logging.h>

namespace epi
{

Eigen::Ref<Eigen::VectorXd> OdeIntegrator::advance(double tmax)
{
    const double t0 = m_result.get_time(m_result.get_num_time_points() - 1);
    assert(tmax > t0);

    const size_t nb_steps = (int)(ceil((tmax - t0) / m_dt)); // estimated number of time steps (if equidistant)

    m_result.reserve(m_result.get_num_time_points() + nb_steps);

    bool step_okay = true;

    double t = t0;
    size_t i = m_result.get_num_time_points() - 1;
    while (std::abs((tmax - t) / (tmax - t0)) > 1e-10) {
        //we don't make timesteps too small as the error estimator of an adaptive integrator
        //may not be able to handle it. this is very conservative and maybe unnecessary,
        //but also unlikely to happen. may need to be reevaluated

        auto dt_eff = std::min(m_dt, tmax - t);
        m_result.add_time_point();
        step_okay &= m_core->step(m_f, m_result[i], t, dt_eff, m_result[i + 1]);
        m_result.get_last_time() = t;

        ++i;

        if (std::abs((tmax - t) / (tmax - t0)) > 1e-10 || dt_eff > m_dt) {
            //store dt only if it's not the last step as it is probably smaller than required for tolerances
            //except if the step function returns a bigger step size so as to not lose efficiency
            m_dt = dt_eff;
        }
    }

    if (!step_okay) {
        log_warning("Adaptive step sizing failed.");
    }
    else if (std::abs((tmax - t) / (tmax - t0)) > 1e-15) {
        log_warning("Last time step too small. Could not reach tmax exactly.");
    }
    else {
        log_info("Adaptive step sizing successful to tolerances.");
    }

    return m_result.get_last_value();
}

} // namespace epi