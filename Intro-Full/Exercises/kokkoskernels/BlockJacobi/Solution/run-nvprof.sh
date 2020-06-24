# FYI
nvprof --query-metrics |& tee log-query-metrics

# check grid and blocksize setting 
nvprof --print-gpu-trace ./blockjacobi |& tee log-trace

# see how team size impact the kernel performance
nvprof -m  \
    achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
l2_utilization,\
gld_throughput,\
gst_throughput,\
l2_read_throughput,\
l2_write_throughput ./blockjacobi -Task 1 -TeamSize 0 |& tee log-analysis-task-1-team-auto

nvprof -m  \
    achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
l2_utilization,\
gld_throughput,\
gst_throughput,\
l2_read_throughput,\
l2_write_throughput ./blockjacobi -Task 2 -TeamSize 0 |& tee log-analysis-task-2-team-auto

nvprof -m  \
    achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
l2_utilization,\
gld_throughput,\
gst_throughput,\
l2_read_throughput,\
l2_write_throughput ./blockjacobi -Task 1 -TeamSize 32 |& tee log-analysis-task-1-team-32

nvprof -m  \
    achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
l2_utilization,\
gld_throughput,\
gst_throughput,\
l2_read_throughput,\
l2_write_throughput ./blockjacobi -Task 2 -TeamSize 32 |& tee log-analysis-task-2-team-32
