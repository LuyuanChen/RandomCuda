#pragma once
#define BLOCK_SIZE		32
#define TILE_SIZE		BLOCK_SIZE
#define TENSOR_SIZE		1024
#define MEMSET_VAL		2.1
#define MUL

#define OPT_TILE_SIZE		16
#define OPT_WIDTH			64	// keep same as the thread number
#define OPT_BLOCK_X_SIZE	OPT_TILE_SIZE
#define OPT_BLOCK_Y_SIZE	OPT_WIDTH / OPT_BLOCK_X_SIZE
	
#define __CUDACC__
