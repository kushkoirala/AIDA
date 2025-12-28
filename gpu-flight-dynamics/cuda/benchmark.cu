/**
 * @file benchmark.cu
 * @brief Performance Benchmarks for GPU Flight Dynamics Simulator
 * 
 * Compares GPU vs CPU performance. This is the key portfolio demonstration.
 * 
 * @author Kushal Koirala
 * @date 2025
 */

#include "flight_dynamics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * TIMING
 * ============================================================================ */

typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

void timer_start(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_stop_ms(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    return (t->end.tv_sec - t->start.tv_sec) * 1000.0 +
           (t->end.tv_nsec - t->start.tv_nsec) / 1000000.0;
}

/* ============================================================================
 * CPU BASELINE
 * ============================================================================ */

void cpu_step(float* state, const float* control, const AircraftParams* params, float dt)
{
    float u = state[FD_U], v = state[FD_V], w = state[FD_W];
    float phi = state[FD_PHI], theta = state[FD_THETA], psi = state[FD_PSI];
    float p = state[FD_P], q = state[FD_Q], r = state[FD_R];
    float z = state[FD_Z];
    
    // Atmosphere
    float altitude = -z;
    float T = 288.15f - 0.0065f * fmaxf(0.0f, altitude);
    float rho = 1.225f * powf(T / 288.15f, 4.256f);
    
    // Airspeed
    float V = sqrtf(u*u + v*v + w*w);
    V = fmaxf(V, 0.1f);
    float alpha = atan2f(w, u);
    float beta = asinf(fmaxf(-1.0f, fminf(1.0f, v / V)));
    
    float qbar = 0.5f * rho * V * V;
    float phat = p * params->geom.b / (2.0f * V);
    float qhat = q * params->geom.c / (2.0f * V);
    float rhat = r * params->geom.b / (2.0f * V);
    
    float da = control[FD_AILERON], de = control[FD_ELEVATOR], dr = control[FD_RUDDER];
    float throttle = control[FD_THROTTLE];
    
    // Aero coefficients
    float CL = params->longi.CL0 + params->longi.CLa * alpha + 
               params->longi.CLq * qhat + params->longi.CLde * de;
    float CD = params->longi.CD0 + params->longi.K * CL * CL;
    float Cm = params->longi.Cm0 + params->longi.Cma * alpha + 
               params->longi.Cmq * qhat + params->longi.Cmde * de;
    float CY = params->latdi.CYb * beta + params->latdi.CYp * phat + 
               params->latdi.CYr * rhat + params->latdi.CYdr * dr;
    float Cl = params->latdi.Clb * beta + params->latdi.Clp * phat + 
               params->latdi.Clr * rhat + params->latdi.Clda * da;
    float Cn = params->latdi.Cnb * beta + params->latdi.Cnp * phat + 
               params->latdi.Cnr * rhat + params->latdi.Cndr * dr;
    
    // Forces
    float L_aero = qbar * params->geom.S * CL;
    float D_aero = qbar * params->geom.S * CD;
    float Y_aero = qbar * params->geom.S * CY;
    
    float cos_a = cosf(alpha), sin_a = sinf(alpha);
    float Fx = -D_aero * cos_a + L_aero * sin_a + throttle * params->prop.thrust_max;
    float Fy = Y_aero;
    float Fz = -D_aero * sin_a - L_aero * cos_a;
    
    // Moments
    float L_mom = qbar * params->geom.S * params->geom.b * Cl;
    float M_mom = qbar * params->geom.S * params->geom.c * Cm;
    float N_mom = qbar * params->geom.S * params->geom.b * Cn;
    
    // Gravity
    float g = 9.80665f;
    float gx = -g * sinf(theta);
    float gy = g * cosf(theta) * sinf(phi);
    float gz = g * cosf(theta) * cosf(phi);
    
    // Accelerations
    float inv_mass = 1.0f / params->mass.mass;
    float u_dot = Fx * inv_mass - q*w + r*v + gx;
    float v_dot = Fy * inv_mass - r*u + p*w + gy;
    float w_dot = Fz * inv_mass - p*v + q*u + gz;
    
    float p_dot = L_mom / params->mass.Ixx;
    float q_dot = M_mom / params->mass.Iyy;
    float r_dot = N_mom / params->mass.Izz;
    
    // Euler rates
    float sin_phi = sinf(phi), cos_phi = cosf(phi);
    float cos_theta = cosf(theta), tan_theta = tanf(theta);
    
    float phi_dot = p + (q*sin_phi + r*cos_phi) * tan_theta;
    float theta_dot = q*cos_phi - r*sin_phi;
    float psi_dot = (q*sin_phi + r*cos_phi) / fmaxf(fabsf(cos_theta), 0.001f);
    
    // Position rates
    float sin_theta = sinf(theta);
    float cos_psi = cosf(psi), sin_psi = sinf(psi);
    
    float x_dot = cos_theta*cos_psi*u + (sin_phi*sin_theta*cos_psi - cos_phi*sin_psi)*v +
                  (cos_phi*sin_theta*cos_psi + sin_phi*sin_psi)*w;
    float y_dot = cos_theta*sin_psi*u + (sin_phi*sin_theta*sin_psi + cos_phi*cos_psi)*v +
                  (cos_phi*sin_theta*sin_psi - sin_phi*cos_psi)*w;
    float z_dot = -sin_theta*u + sin_phi*cos_theta*v + cos_phi*cos_theta*w;
    
    // Integrate
    state[FD_X] += x_dot * dt;
    state[FD_Y] += y_dot * dt;
    state[FD_Z] += z_dot * dt;
    state[FD_U] += u_dot * dt;
    state[FD_V] += v_dot * dt;
    state[FD_W] += w_dot * dt;
    state[FD_PHI] += phi_dot * dt;
    state[FD_THETA] += theta_dot * dt;
    state[FD_PSI] += psi_dot * dt;
    state[FD_P] += p_dot * dt;
    state[FD_Q] += q_dot * dt;
    state[FD_R] += r_dot * dt;
}

void cpu_simulate(float* states, const float* controls, const AircraftParams* params,
                  float dt, int n_instances, int n_steps)
{
    for (int step = 0; step < n_steps; step++) {
        for (int i = 0; i < n_instances; i++) {
            cpu_step(&states[i * FD_STATE_DIM], &controls[i * FD_CONTROL_DIM], params, dt);
        }
    }
}

/* ============================================================================
 * BENCHMARK
 * ============================================================================ */

typedef struct {
    int n_instances;
    int n_steps;
    double cpu_ms;
    double gpu_ms;
    double speedup;
} BenchResult;

void run_benchmark(int n_instances, int n_steps, BenchResult* result)
{
    result->n_instances = n_instances;
    result->n_steps = n_steps;
    
    AircraftParams params;
    fd_default_params(&params);
    float dt = 0.01f;
    
    // Trim
    float trim_state[FD_STATE_DIM];
    float trim_control[FD_CONTROL_DIM];
    fd_compute_trim(&params, 50.0f, 1000.0f, trim_state, trim_control);
    
    // Allocate CPU arrays
    float* cpu_states = (float*)malloc(n_instances * FD_STATE_DIM * sizeof(float));
    float* cpu_controls = (float*)malloc(n_instances * FD_CONTROL_DIM * sizeof(float));
    
    for (int i = 0; i < n_instances; i++) {
        memcpy(&cpu_states[i * FD_STATE_DIM], trim_state, FD_STATE_DIM * sizeof(float));
        memcpy(&cpu_controls[i * FD_CONTROL_DIM], trim_control, FD_CONTROL_DIM * sizeof(float));
        cpu_controls[i * FD_CONTROL_DIM + FD_AILERON] = 0.1f * sinf(i * 0.1f);
    }
    
    // CPU benchmark
    Timer timer;
    timer_start(&timer);
    cpu_simulate(cpu_states, cpu_controls, &params, dt, n_instances, n_steps);
    result->cpu_ms = timer_stop_ms(&timer);
    
    // GPU benchmark
    SimConfig config;
    fd_default_config(&config, n_instances);
    config.dt = dt;
    config.integration_method = 0;  // Euler for fair comparison
    
    SimContext ctx;
    fd_init(&ctx, &config, &params);
    fd_reset(&ctx, trim_state, trim_control);
    
    float* gpu_controls = (float*)malloc(n_instances * FD_CONTROL_DIM * sizeof(float));
    for (int i = 0; i < n_instances; i++) {
        memcpy(&gpu_controls[i * FD_CONTROL_DIM], trim_control, FD_CONTROL_DIM * sizeof(float));
        gpu_controls[i * FD_CONTROL_DIM + FD_AILERON] = 0.1f * sinf(i * 0.1f);
    }
    fd_set_controls(&ctx, gpu_controls);
    
    // Warm-up
    fd_step_n(&ctx, 10);
    cudaDeviceSynchronize();
    fd_reset(&ctx, trim_state, trim_control);
    fd_set_controls(&ctx, gpu_controls);
    
    timer_start(&timer);
    fd_step_n(&ctx, n_steps);
    cudaDeviceSynchronize();
    result->gpu_ms = timer_stop_ms(&timer);
    
    result->speedup = result->cpu_ms / result->gpu_ms;
    
    free(cpu_states);
    free(cpu_controls);
    free(gpu_controls);
    fd_cleanup(&ctx);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main()
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║   GPU-Accelerated Flight Dynamics Simulator - Benchmarks         ║\n");
    printf("║   Author: Kushal Koirala                                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    // GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Benchmarks
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("Benchmark: 1000 simulation steps\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    
    printf("%-12s %12s %12s %10s\n", "Instances", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("─────────────────────────────────────────────────────────────\n");
    
    int counts[] = {10, 100, 500, 1000, 2000, 5000, 10000};
    int n_tests = sizeof(counts) / sizeof(counts[0]);
    
    BenchResult results[20];
    double max_speedup = 0;
    
    for (int i = 0; i < n_tests; i++) {
        printf("Running %d instances...\r", counts[i]);
        fflush(stdout);
        
        run_benchmark(counts[i], 1000, &results[i]);
        
        printf("%-12d %12.2f %12.2f %9.1fx\n",
               results[i].n_instances, results[i].cpu_ms, results[i].gpu_ms, results[i].speedup);
        
        if (results[i].speedup > max_speedup) {
            max_speedup = results[i].speedup;
        }
    }
    
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    
    printf("🚀 Maximum Speedup: %.1fx\n", max_speedup);
    printf("📊 GPU Throughput:  %.2f million sim-steps/sec\n",
           (results[n_tests-1].n_instances * 1000.0) / results[n_tests-1].gpu_ms / 1000.0);
    
    // Markdown table
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("Markdown Table (for README/Portfolio):\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    
    printf("| Instances | CPU (ms) | GPU (ms) | Speedup |\n");
    printf("|----------:|---------:|---------:|--------:|\n");
    for (int i = 0; i < n_tests; i++) {
        printf("| %9d | %8.2f | %8.2f | %6.1fx |\n",
               results[i].n_instances, results[i].cpu_ms, results[i].gpu_ms, results[i].speedup);
    }
    printf("\n");
    
    return 0;
}
