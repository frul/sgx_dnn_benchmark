#pragma once

#include "../common/common.hpp"

class Routine
{
public:
    virtual int execute() = 0;
    virtual std::string getName() = 0;
    virtual ~Routine() {};
};

class InferenceRoutine : public Routine
{
public:
    int execute() override {
        cnn_inference_f32_cpp_routine(parse_engine_kind(1, NULL));
        return 0;
    }

    std::string getName() override {
        return "Inference on CPU";
    }
};

class TrainingRoutine : public Routine
{
public:
    int execute() override {
        cnn_training_f32_cpp_routine(parse_engine_kind(1, NULL));
        return 0;
    }

    std::string getName() override {
        return "Training on CPU";
    }
};
