/**
 * @file flight_sim_kernels.cu
 * @brief Consolidated GPU Flight Dynamics Kernels
 * 
 * This file contains all CUDA device functions and kernels in a single
 * compilation unit to avoid linker issues with device functions.
 * 
 * @author Kushal Koirala
 * @date 2025
 * 
 * NVIDIA Stack: CUDA 12.x, RTX 4060 (sm_89)
 */

#include "flight_dynamics.h"
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define PI_F 3.14159265358979323846f

/* ============================================================================
 * ATMOSPHERE MODEL (Device Functions)
 * ============================================================================ */

/**
 * @brief Compute atmospheric properties at given altitude
 */
__device__ void atmosphere_isa(float altitude, float* rho, float* temp, float* pressure)
{
    float h = fmaxf(0.0f, altitude);
    
    float T, P, density;
    
    if (h <= 11000.0f) {
        // Troposphere
        T = FD_TEMP_SL - FD_LAPSE_RATE * h;
        float temp_ratio = T / FD_TEMP_SL;
        float exponent = FD_GRAVITY / (FD_LAPSE_RATE * FD_GAS_CONST);
        
        P = FD_PRESS_SL * powf(temp_ratio, exponent);
        density = FD_RHO_SL * powf(temp_ratio, exponent - 1.0f);
    }
    else if (h <= 20000.0f) {
        // Stratosphere (isothermal)
        float T_11k = 216.65f;
        float temp_ratio_11k = T_11k / FD_TEMP_SL;
        float exponent = FD_GRAVITY / (FD_LAPSE_RATE * FD_GAS_CONST);
        
        float P_11k = FD_PRESS_SL * powf(temp_ratio_11k, exponent);
        float rho_11k = FD_RHO_SL * powf(temp_ratio_11k, exponent - 1.0f);
        
        float dh = h - 11000.0f;
        float scale_height = FD_GAS_CONST * T_11k / FD_GRAVITY;
        float decay = expf(-dh / scale_height);
        
        T = T_11k;
        P = P_11k * decay;
        density = rho_11k * decay;
    }
    else {
        // Above 20km (use 20km values)
        float T_11k = 216.65f;
        float temp_ratio_11k = T_11k / FD_TEMP_SL;
        float exponent = FD_GRAVITY / (FD_LAPSE_RATE * FD_GAS_CONST);
        
        float P_11k = FD_PRESS_SL * powf(temp_ratio_11k, exponent);
        float rho_11k = FD_RHO_SL * powf(temp_ratio_11k, exponent - 1.0f);
        
        float scale_height = FD_GAS_CONST * T_11k / FD_GRAVITY;
        float decay = expf(-9000.0f / scale_height);
        
        T = T_11k;
        P = P_11k * decay;
        density = rho_11k * decay;
    }
    
    *rho = density;
    *temp = T;
    if (pressure != NULL) {
        *pressure = P;
    }
}

/* ============================================================================
 * AERODYNAMICS (Device Functions)
 * ============================================================================ */

__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

__device__ __forceinline__ float safe_asinf(float x)
{
    return asinf(clampf(x, -1.0f, 1.0f));
}

__device__ float compute_aero_angles(float u, float v, float w,
                                     float* alpha, float* beta)
{
    float V = sqrtf(u*u + v*v + w*w);
    V = fmaxf(V, 0.1f);
    
    *alpha = atan2f(w, u);
    *beta = safe_asinf(v / V);
    
    return V;
}

/**
 * @brief Compute aerodynamic forces and moments
 */
__device__ void compute_aerodynamics(
    const float* state,
    const float* control,
    const AircraftParams* params,
    float rho,
    int enable_stall,
    float* forces,
    float* moments)
{
    // Extract velocities and rates
    float u = state[FD_U];
    float v = state[FD_V];
    float w = state[FD_W];
    float p = state[FD_P];
    float q = state[FD_Q];
    float r = state[FD_R];
    
    // Controls
    float da = control[FD_AILERON];
    float de = control[FD_ELEVATOR];
    float dr = control[FD_RUDDER];
    
    // Airspeed and angles
    float alpha, beta;
    float V = compute_aero_angles(u, v, w, &alpha, &beta);
    
    // Dynamic pressure
    float qbar = 0.5f * rho * V * V;
    
    float S = params->geom.S;
    float b = params->geom.b;
    float c = params->geom.c;
    
    // Non-dimensional rates
    float phat = p * b / (2.0f * V);
    float qhat = q * c / (2.0f * V);
    float rhat = r * b / (2.0f * V);
    
    // Lift coefficient
    float CL = params->longi.CL0 
             + params->longi.CLa * alpha
             + params->longi.CLq * qhat
             + params->longi.CLde * de;
    
    if (enable_stall) {
        CL = clampf(CL, params->longi.CLmin, params->longi.CLmax);
    }
    
    // Drag coefficient
    float CD = params->longi.CD0 
             + params->longi.K * CL * CL
             + params->longi.CDa * fabsf(alpha);
    
    // Pitch moment
    float Cm = params->longi.Cm0
             + params->longi.Cma * alpha
             + params->longi.Cmq * qhat
             + params->longi.Cmde * de;
    
    // Side force
    float CY = params->latdi.CYb * beta
             + params->latdi.CYp * phat
             + params->latdi.CYr * rhat
             + params->latdi.CYda * da
             + params->latdi.CYdr * dr;
    
    // Roll moment
    float Cl = params->latdi.Clb * beta
             + params->latdi.Clp * phat
             + params->latdi.Clr * rhat
             + params->latdi.Clda * da
             + params->latdi.Cldr * dr;
    
    // Yaw moment
    float Cn = params->latdi.Cnb * beta
             + params->latdi.Cnp * phat
             + params->latdi.Cnr * rhat
             + params->latdi.Cnda * da
             + params->latdi.Cndr * dr;
    
    // Convert to forces
    float L_aero = qbar * S * CL;
    float D_aero = qbar * S * CD;
    float Y_aero = qbar * S * CY;
    
    // Rotate from wind to body frame
    float cos_a = cosf(alpha);
    float sin_a = sinf(alpha);
    
    forces[0] = -D_aero * cos_a + L_aero * sin_a;
    forces[1] = Y_aero;
    forces[2] = -D_aero * sin_a - L_aero * cos_a;
    
    // Moments
    moments[0] = qbar * S * b * Cl;
    moments[1] = qbar * S * c * Cm;
    moments[2] = qbar * S * b * Cn;
}

__device__ float compute_thrust(float throttle, const PropulsionParams* params)
{
    float thr = clampf(throttle, 0.0f, 1.0f);
    return params->thrust_min + thr * (params->thrust_max - params->thrust_min);
}

__device__ void gravity_body_frame(float phi, float theta,
                                   float* gx, float* gy, float* gz)
{
    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);
    
    *gx = -FD_GRAVITY * sin_theta;
    *gy =  FD_GRAVITY * cos_theta * sin_phi;
    *gz =  FD_GRAVITY * cos_theta * cos_phi;
}

/* ============================================================================
 * EQUATIONS OF MOTION (Device Functions)
 * ============================================================================ */

__device__ __forceinline__ float normalize_angle(float angle)
{
    while (angle > PI_F) angle -= 2.0f * PI_F;
    while (angle < -PI_F) angle += 2.0f * PI_F;
    return angle;
}

__device__ void compute_gamma_terms(const MassProperties* mass,
                                    float* G1, float* G2, float* G3, float* G4,
                                    float* G5, float* G6, float* G7, float* G8)
{
    float Ixx = mass->Ixx;
    float Iyy = mass->Iyy;
    float Izz = mass->Izz;
    float Ixz = mass->Ixz;
    
    float Gamma = Ixx * Izz - Ixz * Ixz;
    float inv_Gamma = 1.0f / Gamma;
    float inv_Iyy = 1.0f / Iyy;
    
    *G1 = (Ixz * (Ixx - Iyy + Izz)) * inv_Gamma;
    *G2 = (Izz * (Izz - Iyy) + Ixz * Ixz) * inv_Gamma;
    *G3 = Izz * inv_Gamma;
    *G4 = Ixz * inv_Gamma;
    *G5 = (Izz - Ixx) * inv_Iyy;
    *G6 = Ixz * inv_Iyy;
    *G7 = ((Ixx - Iyy) * Ixx + Ixz * Ixz) * inv_Gamma;
    *G8 = Ixx * inv_Gamma;
}

/**
 * @brief Compute state derivatives
 */
__device__ void compute_state_derivatives(
    const float* state,
    const float* control,
    const AircraftParams* params,
    int enable_stall,
    float* state_dot)
{
    float z = state[FD_Z];
    float u = state[FD_U];
    float v = state[FD_V];
    float w = state[FD_W];
    float phi = state[FD_PHI];
    float theta = state[FD_THETA];
    float psi = state[FD_PSI];
    float p = state[FD_P];
    float q = state[FD_Q];
    float r = state[FD_R];
    
    float throttle = control[FD_THROTTLE];
    
    // Atmosphere
    float altitude = -z;
    float rho, temp;
    atmosphere_isa(altitude, &rho, &temp, NULL);
    
    // Aerodynamics
    float aero_forces[3], aero_moments[3];
    compute_aerodynamics(state, control, params, rho, enable_stall,
                        aero_forces, aero_moments);
    
    // Thrust
    float thrust = compute_thrust(throttle, &params->prop);
    
    // Gravity
    float gx, gy, gz;
    gravity_body_frame(phi, theta, &gx, &gy, &gz);
    
    // Total forces
    float Fx = aero_forces[0] + thrust;
    float Fy = aero_forces[1];
    float Fz = aero_forces[2];
    
    float L = aero_moments[0];
    float M = aero_moments[1];
    float N = aero_moments[2];
    
    // Translational dynamics
    float inv_mass = 1.0f / params->mass.mass;
    
    float u_dot = Fx * inv_mass - q*w + r*v + gx;
    float v_dot = Fy * inv_mass - r*u + p*w + gy;
    float w_dot = Fz * inv_mass - p*v + q*u + gz;
    
    // Rotational dynamics
    float G1, G2, G3, G4, G5, G6, G7, G8;
    compute_gamma_terms(&params->mass, &G1, &G2, &G3, &G4, &G5, &G6, &G7, &G8);
    
    float p_dot = G1*p*q - G2*q*r + G3*L + G4*N;
    float q_dot = G5*p*r - G6*(p*p - r*r) + M / params->mass.Iyy;
    float r_dot = G7*p*q - G1*q*r + G4*L + G8*N;
    
    // Kinematic equations
    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    float cos_theta = cosf(theta);
    float tan_theta = tanf(theta);
    
    float sec_theta = 1.0f / fmaxf(fabsf(cos_theta), 0.001f);
    if (cos_theta < 0.0f) sec_theta = -sec_theta;
    
    float phi_dot = p + (q*sin_phi + r*cos_phi) * tan_theta;
    float theta_dot = q*cos_phi - r*sin_phi;
    float psi_dot = (q*sin_phi + r*cos_phi) * sec_theta;
    
    // Navigation equations
    float sin_theta = sinf(theta);
    float cos_psi = cosf(psi);
    float sin_psi = sinf(psi);
    
    float R11 = cos_theta * cos_psi;
    float R12 = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi;
    float R13 = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi;
    float R21 = cos_theta * sin_psi;
    float R22 = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi;
    float R23 = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi;
    float R31 = -sin_theta;
    float R32 = sin_phi * cos_theta;
    float R33 = cos_phi * cos_theta;
    
    float x_dot = R11*u + R12*v + R13*w;
    float y_dot = R21*u + R22*v + R23*w;
    float z_dot = R31*u + R32*v + R33*w;
    
    // Store derivatives
    state_dot[FD_X] = x_dot;
    state_dot[FD_Y] = y_dot;
    state_dot[FD_Z] = z_dot;
    state_dot[FD_U] = u_dot;
    state_dot[FD_V] = v_dot;
    state_dot[FD_W] = w_dot;
    state_dot[FD_PHI] = phi_dot;
    state_dot[FD_THETA] = theta_dot;
    state_dot[FD_PSI] = psi_dot;
    state_dot[FD_P] = p_dot;
    state_dot[FD_Q] = q_dot;
    state_dot[FD_R] = r_dot;
}

/* ============================================================================
 * INTEGRATION KERNELS
 * ============================================================================ */

__global__ void integrate_euler_kernel(
    float* states,
    const float* controls,
    const AircraftParams* params,
    float dt,
    int n_instances,
    int enable_stall)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_instances) return;
    
    float* state = &states[idx * FD_STATE_DIM];
    const float* control = &controls[idx * FD_CONTROL_DIM];
    
    float state_dot[FD_STATE_DIM];
    compute_state_derivatives(state, control, params, enable_stall, state_dot);
    
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state[i] += dt * state_dot[i];
    }
    
    state[FD_PHI] = normalize_angle(state[FD_PHI]);
    state[FD_THETA] = normalize_angle(state[FD_THETA]);
    state[FD_PSI] = normalize_angle(state[FD_PSI]);
}

__global__ void integrate_rk4_kernel(
    float* states,
    const float* controls,
    const AircraftParams* params,
    float dt,
    int n_instances,
    int enable_stall)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_instances) return;
    
    float* state = &states[idx * FD_STATE_DIM];
    const float* control = &controls[idx * FD_CONTROL_DIM];
    
    float state_orig[FD_STATE_DIM];
    float state_temp[FD_STATE_DIM];
    
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state_orig[i] = state[i];
    }
    
    // k1
    float k1[FD_STATE_DIM];
    compute_state_derivatives(state_orig, control, params, enable_stall, k1);
    
    // k2
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state_temp[i] = state_orig[i] + 0.5f * dt * k1[i];
    }
    float k2[FD_STATE_DIM];
    compute_state_derivatives(state_temp, control, params, enable_stall, k2);
    
    // k3
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state_temp[i] = state_orig[i] + 0.5f * dt * k2[i];
    }
    float k3[FD_STATE_DIM];
    compute_state_derivatives(state_temp, control, params, enable_stall, k3);
    
    // k4
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state_temp[i] = state_orig[i] + dt * k3[i];
    }
    float k4[FD_STATE_DIM];
    compute_state_derivatives(state_temp, control, params, enable_stall, k4);
    
    // Final update
    float dt_6 = dt / 6.0f;
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state[i] = state_orig[i] + dt_6 * (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }
    
    state[FD_PHI] = normalize_angle(state[FD_PHI]);
    state[FD_THETA] = normalize_angle(state[FD_THETA]);
    state[FD_PSI] = normalize_angle(state[FD_PSI]);
}

/* ============================================================================
 * STATE INITIALIZATION KERNEL
 * ============================================================================ */

__global__ void init_states_kernel(
    float* states,
    const float* initial_state,
    int n_instances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_instances) return;
    
    float* state = &states[idx * FD_STATE_DIM];
    
    #pragma unroll
    for (int i = 0; i < FD_STATE_DIM; i++) {
        state[i] = initial_state[i];
    }
}

/* ============================================================================
 * HOST API IMPLEMENTATION
 * ============================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

void fd_default_params(AircraftParams* params)
{
    // Mass properties (Cessna 172-like)
    params->mass.mass = 1043.0f;
    params->mass.Ixx = 1285.0f;
    params->mass.Iyy = 1825.0f;
    params->mass.Izz = 2667.0f;
    params->mass.Ixz = 0.0f;
    
    // Geometry
    params->geom.S = 16.17f;
    params->geom.b = 10.92f;
    params->geom.c = 1.49f;
    params->geom.AR = params->geom.b * params->geom.b / params->geom.S;
    params->geom.e = 0.8f;
    
    // Longitudinal
    params->longi.CL0 = 0.307f;
    params->longi.CLa = 4.41f;
    params->longi.CLq = 3.9f;
    params->longi.CLde = 0.43f;
    params->longi.CLmax = 1.4f;
    params->longi.CLmin = -0.5f;
    params->longi.CD0 = 0.027f;
    params->longi.K = 1.0f / (FD_PI * params->geom.AR * params->geom.e);
    params->longi.CDa = 0.0f;
    params->longi.Cm0 = 0.04f;
    params->longi.Cma = -0.613f;
    params->longi.Cmq = -12.4f;
    params->longi.Cmde = -1.122f;
    
    // Lateral
    params->latdi.CYb = -0.393f;
    params->latdi.CYp = -0.075f;
    params->latdi.CYr = 0.214f;
    params->latdi.CYda = 0.0f;
    params->latdi.CYdr = 0.187f;
    params->latdi.Clb = -0.0923f;
    params->latdi.Clp = -0.484f;
    params->latdi.Clr = 0.0798f;
    params->latdi.Clda = 0.229f;
    params->latdi.Cldr = 0.0147f;
    params->latdi.Cnb = 0.0587f;
    params->latdi.Cnp = -0.0278f;
    params->latdi.Cnr = -0.0937f;
    params->latdi.Cnda = -0.0216f;
    params->latdi.Cndr = -0.0645f;
    
    // Propulsion
    params->prop.thrust_max = 2500.0f;
    params->prop.thrust_min = 50.0f;
    params->prop.tau = 0.5f;
}

void fd_default_config(SimConfig* config, int n_instances)
{
    config->n_instances = n_instances;
    config->dt = 0.01f;
    config->integration_method = 2;
    config->enable_stall = 1;
    config->enable_ground = 0;
    config->ground_altitude = 0.0f;
}

int fd_init(SimContext* ctx, const SimConfig* config, const AircraftParams* params)
{
    ctx->config = *config;
    ctx->sim_time = 0.0f;
    
    cudaError_t err;
    
    err = cudaGetDevice(&ctx->device_id);
    if (err != cudaSuccess) return -1;
    
    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) return -1;
    
    int n = config->n_instances;
    size_t state_size = n * FD_STATE_DIM * sizeof(float);
    size_t control_size = n * FD_CONTROL_DIM * sizeof(float);
    
    err = cudaMalloc(&ctx->d_states, state_size);
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&ctx->d_controls, control_size);
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&ctx->d_state_dots, state_size);
    if (err != cudaSuccess) return -1;
    
    err = cudaMemset(ctx->d_states, 0, state_size);
    if (err != cudaSuccess) return -1;
    
    err = cudaMemset(ctx->d_controls, 0, control_size);
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&ctx->d_params, sizeof(AircraftParams));
    if (err != cudaSuccess) return -1;
    
    err = cudaMemcpy(ctx->d_params, params, sizeof(AircraftParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    
    printf("FlightDynamics: Initialized %d instances on GPU %d\n", n, ctx->device_id);
    
    return 0;
}

void fd_cleanup(SimContext* ctx)
{
    if (ctx->d_states) cudaFree(ctx->d_states);
    if (ctx->d_controls) cudaFree(ctx->d_controls);
    if (ctx->d_state_dots) cudaFree(ctx->d_state_dots);
    if (ctx->d_params) cudaFree(ctx->d_params);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    
    ctx->d_states = NULL;
    ctx->d_controls = NULL;
    ctx->d_state_dots = NULL;
    ctx->d_params = NULL;
    ctx->stream = 0;
}

void fd_reset(SimContext* ctx, const float* initial_state, const float* initial_control)
{
    int n = ctx->config.n_instances;
    
    float* d_init_state;
    cudaMalloc(&d_init_state, FD_STATE_DIM * sizeof(float));
    cudaMemcpy(d_init_state, initial_state, FD_STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_states_kernel<<<blocks, threads, 0, ctx->stream>>>(ctx->d_states, d_init_state, n);
    
    cudaFree(d_init_state);
    
    // Broadcast control
    for (int i = 0; i < n; i++) {
        cudaMemcpy(&ctx->d_controls[i * FD_CONTROL_DIM], initial_control,
                   FD_CONTROL_DIM * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ctx->sim_time = 0.0f;
    cudaStreamSynchronize(ctx->stream);
}

void fd_set_controls(SimContext* ctx, const float* controls)
{
    size_t size = ctx->config.n_instances * FD_CONTROL_DIM * sizeof(float);
    cudaMemcpyAsync(ctx->d_controls, controls, size, cudaMemcpyHostToDevice, ctx->stream);
}

void fd_get_states(const SimContext* ctx, float* states)
{
    size_t size = ctx->config.n_instances * FD_STATE_DIM * sizeof(float);
    cudaMemcpy(states, ctx->d_states, size, cudaMemcpyDeviceToHost);
}

float* fd_get_states_device(const SimContext* ctx)
{
    return ctx->d_states;
}

void fd_step(SimContext* ctx)
{
    int n = ctx->config.n_instances;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    if (ctx->config.integration_method == 0) {
        integrate_euler_kernel<<<blocks, threads, 0, ctx->stream>>>(
            ctx->d_states, ctx->d_controls, ctx->d_params,
            ctx->config.dt, n, ctx->config.enable_stall);
    } else {
        integrate_rk4_kernel<<<blocks, threads, 0, ctx->stream>>>(
            ctx->d_states, ctx->d_controls, ctx->d_params,
            ctx->config.dt, n, ctx->config.enable_stall);
    }
    
    ctx->sim_time += ctx->config.dt;
}

void fd_step_n(SimContext* ctx, int n_steps)
{
    for (int i = 0; i < n_steps; i++) {
        fd_step(ctx);
    }
    cudaStreamSynchronize(ctx->stream);
}

float fd_get_time(const SimContext* ctx)
{
    return ctx->sim_time;
}

void fd_atmosphere(float altitude, AtmosphereState* atm)
{
    float h = fmaxf(0.0f, altitude);
    
    if (h <= 11000.0f) {
        atm->temperature = FD_TEMP_SL - FD_LAPSE_RATE * h;
        float temp_ratio = atm->temperature / FD_TEMP_SL;
        float exponent = FD_GRAVITY / (FD_LAPSE_RATE * FD_GAS_CONST);
        atm->pressure = FD_PRESS_SL * powf(temp_ratio, exponent);
        atm->rho = FD_RHO_SL * powf(temp_ratio, exponent - 1.0f);
    } else {
        float T_11k = 216.65f;
        float temp_ratio_11k = T_11k / FD_TEMP_SL;
        float exponent = FD_GRAVITY / (FD_LAPSE_RATE * FD_GAS_CONST);
        float P_11k = FD_PRESS_SL * powf(temp_ratio_11k, exponent);
        float rho_11k = FD_RHO_SL * powf(temp_ratio_11k, exponent - 1.0f);
        float dh = fminf(h - 11000.0f, 9000.0f);
        float scale_height = FD_GAS_CONST * T_11k / FD_GRAVITY;
        float decay = expf(-dh / scale_height);
        atm->temperature = T_11k;
        atm->pressure = P_11k * decay;
        atm->rho = rho_11k * decay;
    }
    
    atm->speed_sound = sqrtf(1.4f * FD_GAS_CONST * atm->temperature);
}

int fd_compute_trim(const AircraftParams* params, float airspeed, float altitude,
                    float* state, float* control)
{
    memset(state, 0, FD_STATE_DIM * sizeof(float));
    memset(control, 0, FD_CONTROL_DIM * sizeof(float));
    
    state[FD_Z] = -altitude;
    
    AtmosphereState atm;
    fd_atmosphere(altitude, &atm);
    
    float weight = params->mass.mass * FD_GRAVITY;
    float qbar = 0.5f * atm.rho * airspeed * airspeed;
    float CL_required = weight / (qbar * params->geom.S);
    
    if (CL_required > params->longi.CLmax || CL_required < params->longi.CLmin) {
        fprintf(stderr, "Trim: Required CL=%.3f out of range\n", CL_required);
        return -1;
    }
    
    float alpha = (CL_required - params->longi.CL0) / params->longi.CLa;
    
    state[FD_U] = airspeed * cosf(alpha);
    state[FD_W] = airspeed * sinf(alpha);
    state[FD_THETA] = alpha;
    
    float Cm_basic = params->longi.Cm0 + params->longi.Cma * alpha;
    control[FD_ELEVATOR] = -Cm_basic / params->longi.Cmde;
    control[FD_ELEVATOR] = fmaxf(-1.0f, fminf(1.0f, control[FD_ELEVATOR]));
    
    float CD = params->longi.CD0 + params->longi.K * CL_required * CL_required;
    float drag = qbar * params->geom.S * CD;
    float thrust_required = drag / cosf(alpha);
    
    control[FD_THROTTLE] = (thrust_required - params->prop.thrust_min) /
                           (params->prop.thrust_max - params->prop.thrust_min);
    control[FD_THROTTLE] = fmaxf(0.0f, fminf(1.0f, control[FD_THROTTLE]));
    
    printf("Trim: V=%.1f m/s, alt=%.0f m, alpha=%.2f deg, de=%.3f, thr=%.3f\n",
           airspeed, altitude, alpha * FD_RAD2DEG, control[FD_ELEVATOR], control[FD_THROTTLE]);
    
    return 0;
}
