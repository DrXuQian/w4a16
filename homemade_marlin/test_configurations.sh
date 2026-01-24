#!/bin/bash
# Test various Marlin configurations

echo "========================================"
echo "Testing Marlin Standalone Configurations"
echo "========================================"
echo ""

echo "Test 1: Default configuration"
./marlin_standalone
echo ""

echo "Test 2: Batch size 1 (inference)"
./marlin_standalone -m 1 -n 4096 -k 4096 -g 128
echo ""

echo "Test 3: Large matrix with per-column quantization"
./marlin_standalone -m 64 -n 8192 -k 8192 -g -1
echo ""

echo "Test 4: Custom number of SMs"
./marlin_standalone -m 32 -n 4096 -k 4096 -g 128 -s 50
echo ""

echo "Test 5: Small matrix"
./marlin_standalone -m 16 -n 256 -k 512 -g 128
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
