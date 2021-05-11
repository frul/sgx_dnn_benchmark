/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example cnn_inference_f32.cpp
/// @copybrief cnn_inference_f32_cpp
/// > Annotated version: @ref cnn_inference_f32_cpp

/// @page cnn_inference_f32_cpp CNN f32 inference example
/// This C++ API example demonstrates how to build an AlexNet neural
/// network topology for forward-pass inference.
///
/// > Example code: @ref cnn_inference_f32.cpp
///
/// Some key take-aways include:
///
/// * How tensors are implemented and submitted to primitives.
/// * How primitives are created.
/// * How primitives are sequentially submitted to the network, where the output
///   from primitives is passed as input to the next primitive. The latter
///   specifies a dependency between the primitive input and output data.
/// * Specific 'inference-only' configurations.
/// * Limiting the number of reorders performed that are detrimental
///   to performance.
///
/// The example implements the AlexNet layers
/// as numbered primitives (for example, conv1, pool1, conv2).

#pragma once

#include <assert.h>

#include <algorithm>
#include <stdlib.h>
#include <string>

#include <cmath>
#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "dnnl.hpp"

using namespace dnnl;

using namespace std;

 memory::dim product(const memory::dims &dims);

 void cnn_inference_f32_cpp_routine(engine::kind engine_kind, int times = 1);

 void cnn_training_f32_cpp_routine(engine::kind engine_kind);

 dnnl::engine::kind parse_engine_kind(
        int argc, char **argv, int extra_args = 0);

 void read_from_dnnl_memory(void *handle, dnnl::memory &mem);

 void write_to_dnnl_memory(void *handle, dnnl::memory &mem);