#!/bin/bash

# Total instances and maximum instances to run concurrently
total_instances=175
max_concurrent_instances=192

# Run instances using GNU parallel with a limit on concurrent jobs
seq 1 "$total_instances" | parallel -j "$max_concurrent_instances" '
    run_instance() {
        num=$1
        output_file="logfile/output_mcmc_$num.txt"  # Unique file name for each instance
        python simulation_mcmc.py --num "$num" > "$output_file" 2>&1
    }

    run_instance {}'


