#ifndef EPI_ABM_LOCKDOWN_RULES_H
#define EPI_ABM_LOCKDOWN_RULES_H

#include "epidemiology/abm/time.h"
#include "epidemiology/abm/location_type.h"
#include "epidemiology/abm/person.h"
#include "epidemiology/abm/parameters.h"

#include "epidemiology/secir/damping.h"
#include "epidemiology/secir/contact_matrix.h"

namespace epi
{

/**
 * LockdownRules implements non phamarceutical interventions via dampings.
 * For interventions, people are randomly divided into groups, e.g. one group works at home and the other group still goes to work.
 * The probability with which a person belongs to a certain group is time dependet. This change
 * in probabilty is implemented by using dampings.
 */

    
/**
 * Persons who are in home office are staying at home instead of going to work.
 * @param t_begin begin of the intervention
 * @param p percentage of people that work in home office
 * @param params migration parameters that include damping
 */
void set_home_office(TimePoint t_begin, double p, AbmMigrationParameters& params);
 
/**
 * If schools are closed, students stay at home instead of going to school.
 * @param t_begin begin of the intervention
 * @param p percentage of people that are homeschooled
 * @param params migration parameters that include damping
 */
void set_school_closure(TimePoint t_begin, double p, AbmMigrationParameters& params);



/** 
 * During lockdown people join social events less often.
 * If a person joins a social event is a random event (exponentially distributed).
 * The damping changes the parameter of the exponential distribution, where a damping of 0 corresponds to no damping
 * and a damping of 1 means that no social events are happening.
 * @param t_begin begin of the intervention
 * @param p damping between 0 and 1 that changes the parameter of the exponential distribution
 * @param params migration parameters that include damping
 */
void close_social_events(TimePoint t_begin, double p, AbmMigrationParameters& params);



} //namespace epi

#endif // LOCKDOWN_RULES_H