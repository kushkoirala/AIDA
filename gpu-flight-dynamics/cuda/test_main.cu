/**
 * @file test_main.cu
 * @brief Test program for GPU Flight Dynamics Simulator
 * 
 * @author Kushal Koirala
 * @date 2025
 */

#include "flight_dynamics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

int tests_run = 0;
int tests_passed = 0;

void test_result(const char* name, int passed) {
    tests_run++;
    if (passed) {
        tests_passed++;
        printf("  [%s] %s\n", TEST_PASS, name);
    } else {
        printf("  [%s] %s\n", TEST_FAIL, name);
    }
}

/* ============================================================================
 * ATMOSPHERE TESTS
 * ============================================================================ */

void test_atmosphere() {
    printf("\n=== Atmosphere Model Tests ===\n");
    
    AtmosphereState atm;
    
    // Test sea level
    fd_atmosphere(0.0f, &atm);
    test_result("Sea level density ~1.225 kg/m³", fabsf(atm.rho - 1.225f) < 0.01f);
    test_result("Sea level temperature ~288 K", fabsf(atm.temperature - 288.15f) < 1.0f);
    
    // Test at 5000m
    fd_atmosphere(5000.0f, &atm);
    test_result("5000m density ~0.736 kg/m³", fabsf(atm.rho - 0.736f) < 0.05f);
    
    // Test at 10000m
    fd_atmosphere(10000.0f, &atm);
    test_result("10000m density ~0.414 kg/m³", fabsf(atm.rho - 0.414f) < 0.05f);
}

/* ============================================================================
 * TRIM TESTS
 * ============================================================================ */

void test_trim() {
    printf("\n=== Trim Computation Tests ===\n");
    
    AircraftParams params;
    fd_default_params(&params);
    
    float state[FD_STATE_DIM];
    float control[FD_CONTROL_DIM];
    
    int result = fd_compute_trim(&params, 50.0f, 1000.0f, state, control);
    
    test_result("Trim found successfully", result == 0);
    test_result("Forward velocity > 0", state[FD_U] > 0.0f);
    test_result("Throttle in valid range", 
                control[FD_THROTTLE] >= 0.0f && control[FD_THROTTLE] <= 1.0f);
    test_result("Elevator in valid range",
                control[FD_ELEVATOR] >= -1.0f && control[FD_ELEVATOR] <= 1.0f);
}

/* ============================================================================
 * SIMULATION TESTS
 * ============================================================================ */

void test_simulation() {
    printf("\n=== Simulation Tests ===\n");
    
    AircraftParams params;
    fd_default_params(&params);
    
    SimConfig config;
    fd_default_config(&config, 100);
    config.dt = 0.01f;
    config.integration_method = 2;  // RK4
    
    SimContext ctx;
    int result = fd_init(&ctx, &config, &params);
    test_result("Initialization successful", result == 0);
    
    if (result != 0) {
        printf("  Skipping remaining simulation tests due to init failure\n");
        return;
    }
    
    // Compute trim
    float trim_state[FD_STATE_DIM];
    float trim_control[FD_CONTROL_DIM];
    fd_compute_trim(&params, 50.0f, 1000.0f, trim_state, trim_control);
    
    // Reset to trim
    fd_reset(&ctx, trim_state, trim_control);
    
    // Get initial states
    float* states = (float*)malloc(100 * FD_STATE_DIM * sizeof(float));
    fd_get_states(&ctx, states);
    
    float initial_altitude = -states[FD_Z];
    test_result("Initial altitude ~1000m", fabsf(initial_altitude - 1000.0f) < 50.0f);
    
    // Run simulation for 1 second
    fd_step_n(&ctx, 100);
    
    // Get final states
    fd_get_states(&ctx, states);
    
    float final_altitude = -states[FD_Z];
    float altitude_change = fabsf(final_altitude - initial_altitude);
    
    test_result("Trim holds altitude (< 100m change)", altitude_change < 100.0f);
    test_result("Simulation time = 1.0s", fabsf(fd_get_time(&ctx) - 1.0f) < 0.001f);
    
    // Test pitch response
    float* controls = (float*)malloc(100 * FD_CONTROL_DIM * sizeof(float));
    for (int i = 0; i < 100; i++) {
        controls[i * FD_CONTROL_DIM + FD_THROTTLE] = trim_control[FD_THROTTLE];
        controls[i * FD_CONTROL_DIM + FD_AILERON] = 0.0f;
        controls[i * FD_CONTROL_DIM + FD_ELEVATOR] = 0.3f;  // Pull back
        controls[i * FD_CONTROL_DIM + FD_RUDDER] = 0.0f;
    }
    
    fd_set_controls(&ctx, controls);
    fd_step_n(&ctx, 100);
    fd_get_states(&ctx, states);
    
    float pitch_after = states[FD_THETA];
    test_result("Pitch increases with elevator", pitch_after > trim_state[FD_THETA]);
    
    free(states);
    free(controls);
    fd_cleanup(&ctx);
}

/* ============================================================================
 * MULTI-INSTANCE TESTS
 * ============================================================================ */

void test_multi_instance() {
    printf("\n=== Multi-Instance Tests ===\n");
    
    AircraftParams params;
    fd_default_params(&params);
    
    SimConfig config;
    fd_default_config(&config, 1000);
    
    SimContext ctx;
    int result = fd_init(&ctx, &config, &params);
    test_result("1000 instance initialization", result == 0);
    
    if (result != 0) {
        printf("  Skipping remaining multi-instance tests\n");
        return;
    }
    
    // Set different controls for different instances
    float* controls = (float*)malloc(1000 * FD_CONTROL_DIM * sizeof(float));
    for (int i = 0; i < 1000; i++) {
        controls[i * FD_CONTROL_DIM + FD_THROTTLE] = 0.5f;
        controls[i * FD_CONTROL_DIM + FD_AILERON] = (float)i / 1000.0f - 0.5f;
        controls[i * FD_CONTROL_DIM + FD_ELEVATOR] = 0.0f;
        controls[i * FD_CONTROL_DIM + FD_RUDDER] = 0.0f;
    }
    
    float trim_state[FD_STATE_DIM];
    float trim_control[FD_CONTROL_DIM];
    fd_compute_trim(&params, 50.0f, 1000.0f, trim_state, trim_control);
    fd_reset(&ctx, trim_state, trim_control);
    
    fd_set_controls(&ctx, controls);
    fd_step_n(&ctx, 100);
    
    float* states = (float*)malloc(1000 * FD_STATE_DIM * sizeof(float));
    fd_get_states(&ctx, states);
    
    float roll_0 = states[0 * FD_STATE_DIM + FD_PHI];
    float roll_500 = states[500 * FD_STATE_DIM + FD_PHI];
    float roll_999 = states[999 * FD_STATE_DIM + FD_PHI];
    
    test_result("Instances diverge with different controls", 
                fabsf(roll_0 - roll_500) > 0.001f || fabsf(roll_500 - roll_999) > 0.001f);
    
    printf("  [INFO] Roll angles: inst[0]=%.4f, inst[500]=%.4f, inst[999]=%.4f rad\n",
           roll_0, roll_500, roll_999);
    
    free(controls);
    free(states);
    fd_cleanup(&ctx);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     GPU-Accelerated Flight Dynamics Simulator Tests          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Show GPU info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("\nGPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    }
    
    test_atmosphere();
    test_trim();
    test_simulation();
    test_multi_instance();
    
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("══════════════════════════════════════════════════════════════\n\n");
    
    return (tests_passed == tests_run) ? 0 : 1;
}
