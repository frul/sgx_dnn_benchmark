#pragma once

#include <chrono>
#include "Routine.h"

void benchmark(Routine* routine) {
    uint64_t steps = 5;
    uint64_t duration = 0;
    std::chrono::steady_clock::time_point begin, end;
    for (uint64_t i = 0; i < steps; ++i) {
        begin = std::chrono::steady_clock::now();
        int retVal = routine->execute();
        end = std::chrono::steady_clock::now();
        if (retVal) {
            std::cout << "Benchmark error" << std::endl;
        }
        else {
            duration +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / steps;
        }
    }
    std::cout << routine->getName() << " time is: " << duration << " [nanoseconds] " << std::endl;
}
