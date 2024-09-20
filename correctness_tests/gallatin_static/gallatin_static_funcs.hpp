#ifndef STATIC_FUNCS
#define STATIC_FUNCS


/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define GALLATIN_DEBUG 1
#define GALLATIN_ERROR_LOG_LENGTH 10

void helper_open_global_log();

int helper_close_global_log();

void log_n_failures(int n_failures);

#endif