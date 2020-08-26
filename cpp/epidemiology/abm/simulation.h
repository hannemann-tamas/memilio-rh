#ifndef EPI_ABM_SIMULATOR_H
#define EPI_ABM_SIMULATOR_H

#include "epidemiology/abm/world.h"
#include "epidemiology/time_series.h"

namespace epi
{

/**
 * run the simulation in discrete steps, evolve the world and report results.
 */
class AbmSimulation
{
    using ResultVector = Eigen::Matrix<int, Eigen::Index(InfectionState::Count), 1>;

public:
    AbmSimulation(double t, World&& world);
    /** run the simulation from the current time to tmax */
    void advance(double tmax);

    const TimeSeries<double>& get_result() const
    {
        return m_result;
    }

private:
    void store_result_at(double t);

    double m_t;
    double m_dt;
    World m_world;
    TimeSeries<double> m_result;
};

} // namespace epi

#endif