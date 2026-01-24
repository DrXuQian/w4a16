#pragma once

// Minimal stub for TORCH_CHECK used in scalar_type.hpp.
#define TORCH_CHECK(cond, ...) do { (void)sizeof(cond); } while (0)
