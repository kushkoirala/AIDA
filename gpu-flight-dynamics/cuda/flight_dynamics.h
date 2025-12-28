/**
 * @file flight_dynamics.h
 * @brief GPU-Accelerated 6-DOF Flight Dynamics Simulator
 * 
 * This header defines the core data structures and API for the CUDA-accelerated
 * flight dynamics simulator. Designed for parallel simulation of thousands of
 * aircraft instances for reinforcement learning and Monte Carlo analysis.
 * 
 * @author Kushal Koirala
 * @date 2025
 * 
 * NVIDIA Stack: CUDA 12.x
 * 
 * Reference Frames:
 *   - NED (North-East-Down): Inertial frame, origin at initial position
 *   - Body: Fixed to aircraft, X forward, Y right, Z down
 *   
 * State Vector [12]:
 *   [0-2]  Position (x, y, z) in NED frame [m]
 *   [3-5]  Velocity (u, v, w) in body frame [m/s]
 *   [6-8]  Euler angles (phi, theta, psi) [rad]
 *   [9-11] Angular rates (p, q, r) in body frame [rad/s]
 *   
 * Control Vector [4]:
 *   [0] Throttle (0 to 1)
 *   [1] Aileron (-1 to 1, positive = right wing down)
 *   [2] Elevator (-1 to 1, positive = nose up)
 *   [3] Rudder (-1 to 1, positive = nose right)
 */

#ifndef FLIGHT_DYNAMICS_H
#define FLIGHT_DYNAMICS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

// Mathematical constants
#define FD_PI           3.14159265358979323846f
#define FD_DEG2RAD      (FD_PI / 180.0f)
#define FD_RAD2DEG      (180.0f / FD_PI)

// Physical constants
#define FD_GRAVITY      9.80665f        // Standard gravity [m/s^2]
#define FD_GAS_CONST    287.05f         // Gas constant for air [J/(kg·K)]

// Sea level ISA conditions
#define FD_RHO_SL       1.225f          // Sea level density [kg/m^3]
#define FD_TEMP_SL      288.15f         // Sea level temperature [K]
#define FD_PRESS_SL     101325.0f       // Sea level pressure [Pa]
#define FD_LAPSE_RATE   0.0065f         // Temperature lapse rate [K/m]

// State vector dimensions
#define FD_STATE_DIM    12
#define FD_CONTROL_DIM  4

// State indices
#define FD_X            0   // Position North [m]
#define FD_Y            1   // Position East [m]
#define FD_Z            2   // Position Down [m] (altitude = -z)
#define FD_U            3   // Body velocity X [m/s]
#define FD_V            4   // Body velocity Y [m/s]
#define FD_W            5   // Body velocity Z [m/s]
#define FD_PHI          6   // Roll angle [rad]
#define FD_THETA        7   // Pitch angle [rad]
#define FD_PSI          8   // Yaw angle [rad]
#define FD_P            9   // Roll rate [rad/s]
#define FD_Q            10  // Pitch rate [rad/s]
#define FD_R            11  // Yaw rate [rad/s]

// Control indices
#define FD_THROTTLE     0
#define FD_AILERON      1
#define FD_ELEVATOR     2
#define FD_RUDDER       3

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Aircraft mass and inertia properties
 */
typedef struct {
    float mass;         // Total mass [kg]
    float Ixx;          // Moment of inertia about X [kg·m²]
    float Iyy;          // Moment of inertia about Y [kg·m²]
    float Izz;          // Moment of inertia about Z [kg·m²]
    float Ixz;          // Product of inertia [kg·m²]
} MassProperties;

/**
 * @brief Wing and control surface geometry
 */
typedef struct {
    float S;            // Wing reference area [m²]
    float b;            // Wingspan [m]
    float c;            // Mean aerodynamic chord [m]
    float AR;           // Aspect ratio (computed: b²/S)
    float e;            // Oswald efficiency factor
} Geometry;

/**
 * @brief Longitudinal aerodynamic derivatives
 * 
 * Lift: CL = CL0 + CLa*alpha + CLq*qhat + CLde*de
 * Drag: CD = CD0 + K*CL² (parabolic polar)
 * Pitch: Cm = Cm0 + Cma*alpha + Cmq*qhat + Cmde*de
 */
typedef struct {
    // Lift derivatives
    float CL0;          // Zero-alpha lift coefficient
    float CLa;          // Lift curve slope [1/rad]
    float CLq;          // Pitch rate derivative
    float CLde;         // Elevator derivative
    float CLmax;        // Maximum lift coefficient (stall)
    float CLmin;        // Minimum lift coefficient
    
    // Drag derivatives
    float CD0;          // Parasitic drag coefficient
    float K;            // Induced drag factor (1/(pi*AR*e))
    float CDa;          // Linear alpha drag term
    
    // Pitch moment derivatives
    float Cm0;          // Zero-alpha pitching moment
    float Cma;          // Static longitudinal stability [1/rad]
    float Cmq;          // Pitch damping derivative
    float Cmde;         // Elevator effectiveness
} LongitudinalDerivatives;

/**
 * @brief Lateral-directional aerodynamic derivatives
 * 
 * Side Force: CY = CYb*beta + CYp*phat + CYr*rhat + CYda*da + CYdr*dr
 * Roll: Cl = Clb*beta + Clp*phat + Clr*rhat + Clda*da + Cldr*dr
 * Yaw: Cn = Cnb*beta + Cnp*phat + Cnr*rhat + Cnda*da + Cndr*dr
 */
typedef struct {
    // Side force derivatives
    float CYb;          // Sideslip derivative [1/rad]
    float CYp;          // Roll rate derivative
    float CYr;          // Yaw rate derivative
    float CYda;         // Aileron derivative
    float CYdr;         // Rudder derivative
    
    // Rolling moment derivatives
    float Clb;          // Dihedral effect [1/rad]
    float Clp;          // Roll damping
    float Clr;          // Roll due to yaw rate
    float Clda;         // Aileron effectiveness
    float Cldr;         // Rudder-roll coupling
    
    // Yawing moment derivatives
    float Cnb;          // Directional stability [1/rad]
    float Cnp;          // Yaw due to roll rate
    float Cnr;          // Yaw damping
    float Cnda;         // Adverse yaw
    float Cndr;         // Rudder effectiveness
} LateralDerivatives;

/**
 * @brief Propulsion system parameters
 */
typedef struct {
    float thrust_max;   // Maximum thrust [N]
    float thrust_min;   // Minimum/idle thrust [N]
    float tau;          // Engine time constant [s] (first-order lag)
} PropulsionParams;

/**
 * @brief Complete aircraft parameters
 */
typedef struct {
    MassProperties mass;
    Geometry geom;
    LongitudinalDerivatives longi;
    LateralDerivatives latdi;
    PropulsionParams prop;
} AircraftParams;

/**
 * @brief Simulation configuration
 */
typedef struct {
    int n_instances;        // Number of parallel simulations
    float dt;               // Time step [s]
    int integration_method; // 0=Euler, 1=RK2, 2=RK4
    int enable_stall;       // Enable stall modeling
    int enable_ground;      // Enable ground collision
    float ground_altitude;  // Ground level [m] (NED: negative = up)
} SimConfig;

/**
 * @brief Atmospheric state at a point
 */
typedef struct {
    float rho;          // Air density [kg/m³]
    float temperature;  // Temperature [K]
    float pressure;     // Pressure [Pa]
    float speed_sound;  // Speed of sound [m/s]
} AtmosphereState;

/**
 * @brief Simulation context (holds device pointers)
 */
typedef struct {
    // Device arrays
    float* d_states;        // [n_instances, STATE_DIM]
    float* d_controls;      // [n_instances, CONTROL_DIM]
    float* d_state_dots;    // [n_instances, STATE_DIM] (for RK4)
    
    // Device parameters (single copy, shared by all instances)
    AircraftParams* d_params;
    
    // Configuration
    SimConfig config;
    
    // Timing
    float sim_time;
    
    // CUDA resources
    cudaStream_t stream;
    int device_id;
} SimContext;

/* ============================================================================
 * API FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize simulation context
 * 
 * @param ctx Pointer to context structure to initialize
 * @param config Simulation configuration
 * @param params Aircraft parameters
 * @return 0 on success, error code otherwise
 */
int fd_init(SimContext* ctx, const SimConfig* config, const AircraftParams* params);

/**
 * @brief Free simulation context and GPU resources
 */
void fd_cleanup(SimContext* ctx);

/**
 * @brief Reset all instances to initial conditions
 * 
 * @param ctx Simulation context
 * @param initial_state Initial state [STATE_DIM] (applied to all instances)
 * @param initial_control Initial control [CONTROL_DIM]
 */
void fd_reset(SimContext* ctx, const float* initial_state, const float* initial_control);

/**
 * @brief Reset specific instances
 * 
 * @param ctx Simulation context
 * @param instance_ids Array of instance indices to reset
 * @param n_reset Number of instances to reset
 * @param initial_states Initial states [n_reset, STATE_DIM]
 */
void fd_reset_instances(SimContext* ctx, const int* instance_ids, int n_reset,
                        const float* initial_states);

/**
 * @brief Set control inputs for all instances
 * 
 * @param ctx Simulation context
 * @param controls Control inputs [n_instances, CONTROL_DIM]
 */
void fd_set_controls(SimContext* ctx, const float* controls);

/**
 * @brief Advance simulation by one time step
 * 
 * @param ctx Simulation context
 */
void fd_step(SimContext* ctx);

/**
 * @brief Advance simulation by multiple time steps
 * 
 * @param ctx Simulation context
 * @param n_steps Number of steps to advance
 */
void fd_step_n(SimContext* ctx, int n_steps);

/**
 * @brief Get current states (copies from device to host)
 * 
 * @param ctx Simulation context
 * @param states Output buffer [n_instances, STATE_DIM]
 */
void fd_get_states(const SimContext* ctx, float* states);

/**
 * @brief Get device pointer to states (for PyTorch interop)
 * 
 * @param ctx Simulation context
 * @return Device pointer to states array
 */
float* fd_get_states_device(const SimContext* ctx);

/**
 * @brief Update aircraft parameters
 * 
 * @param ctx Simulation context
 * @param params New aircraft parameters
 */
void fd_set_params(SimContext* ctx, const AircraftParams* params);

/**
 * @brief Get simulation time
 */
float fd_get_time(const SimContext* ctx);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize default aircraft parameters (general aviation)
 */
void fd_default_params(AircraftParams* params);

/**
 * @brief Initialize default simulation configuration
 */
void fd_default_config(SimConfig* config, int n_instances);

/**
 * @brief Compute trim state for straight and level flight
 * 
 * @param params Aircraft parameters
 * @param airspeed Target airspeed [m/s]
 * @param altitude Target altitude [m]
 * @param state Output trim state [STATE_DIM]
 * @param control Output trim control [CONTROL_DIM]
 * @return 0 on success, -1 if trim not found
 */
int fd_compute_trim(const AircraftParams* params, float airspeed, float altitude,
                    float* state, float* control);

/**
 * @brief Get atmosphere properties at altitude
 * 
 * @param altitude Altitude above sea level [m]
 * @param atm Output atmosphere state
 */
void fd_atmosphere(float altitude, AtmosphereState* atm);

#ifdef __cplusplus
}
#endif

#endif // FLIGHT_DYNAMICS_H
