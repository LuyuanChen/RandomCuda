#pragma once
#define BLOCK_SIZE		32
#define TILE_SIZE		BLOCK_SIZE
#define TENSOR_SIZE		1024
#define MEMSET_VAL		2.1
#define MUL
#ifndef __CUDACC__
#define __CUDACC__
#endif
