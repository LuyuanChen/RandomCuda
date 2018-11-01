#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor.h"

/// <summary>
/// Multiply the float tensor a and b, to produce c.
/// </summary>
/// <param name="a">Input tensor, assume mxk size</param>
/// <param name="b">Input tensor, assume kxn size</param>
/// <param name="c">Output tensor, assume mxn size</param>
/// <returns>cudaError_t</returns>
cudaError_t matmul(float_tensor &a, float_tensor &b, float_tensor &c);
