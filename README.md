# CSCI596_Option-Pricing

## How to run:

#### Single CPU, Single GPU running on multiple Cuda threads and a single CPU thread:
```console
foo@bar:~$ make main
foo@bar:~$ ./main option_type -B barrier_price -K strike_price -N number_of_paths
```

#### Multiple CPU, Multiple GPU running on multiple Cuda threads and multiple CPU threads:
```console
foo@bar:~$ make main_omp
foo@bar:~$ ./main_omp option_type -B barrier_price -K strike_price -N number_of_paths -threads thread_count
```
#### Multiple Node, Multiple CPUs, Multiple GPUs running on multiple Cuda threads and multiple CPU threads
```console
foo@bar:~$ make main_mpi
foo@bar:~$ mpirun -bind-to none -n number_nodes ./main_mpi option_type -B barrier_price -K strike_price -N number_of_paths -threads thread_count
```
## Brief background

This project simulates barrier options, where the pay-off not only depends on the underlying asset's price at maturity but also on whether the underlying hits a price known as the barrier.

We have implemented the following types of barrier options which can be passed as a command line option:
<ul>
<li>"daoc" - Down and Out Call Options</li>
<li>"uaop" - Up and Out Put Options</li>
<li>"uaic" - Up and In Call Options</li>
<li>"daip" - Down and In Put Options</li>
</ul>

We have set the rebate price as 0, but an option to allow a different rebate price can be added in the fututre.

## Presentation
[Link](https://docs.google.com/presentation/d/1jKo4DxYR8iUAGKAsCCW4L0UDvdvlTYFz5IpUdjG1vR8/edit#slide=id.g2628c323073_0_54)


## Methodology

### Architecture of the system we run this on:
We run our program on the CARC High-Performance computing cluster.
The architecture looks something like this:

![Alt text](image.png)
<p>
The computation is divided between nodes, and each node runs a process.
Each node interacts with each other using the Message Passing Interface (MPI)
<p>
Each node has multiple CPU Cores in it, and these cores can run multiple threads for each node process. These threads use OpenMP for interaction and parallelization of threads
<p>
Each node also has accelerated GPU units associated with it where each unit can run multiple CUDA threads.


### How the code works:

We use the Geometric Brownian Motion to simulate the underlying price, and the discretized Euler method version comes down to this:

$$ S_t = S_{t-1}\ +\  \mu S_{t-1} \Delta t\+ \ \sigma  S_{n-1}\Delta W_t  $$

$S_t$: The price of the underlying at time t <br>
$\mu$ : The expected return <br>
$\sigma$ : The expected volatility<br>
$\Delta t$: The time difference between each iteration <br>
$\Delta W_t$: Random number drawn from a distribution with mean 0 and variance $\Delta t$ (Brownian motion component) <br>
<br>

The code generates an array of random elements $\Delta W_t$ and simulates the price motion according to it. 
We have four versions of the code:
<ol>
<li> <p>Simple single, threaded version
<li> <p>Single CPU, Single GPU running on multiple Cuda threads and a single CPU thread
<li><p> Multiple CPU, Multiple GPU running on multiple Cuda threads and multiple CPU threads
<br> Each CPU runs a single CPU thread. Here, we allocate a GPU to every CPU and reduce the result from multiple CUDA threads running on a Single GPU performed on a Single CPU thread.  
<li> <p> Multiple Node, Multiple CPUs, Multiple GPUs running on multiple Cuda threads and multiple CPU threads<br>
Each node runs a version of Multiple CPUs, Multiple GPUs running on multiple Cuda threads, and multiple CPU threads
</ol>

## Charts:
We perform all the tests on Down-and-in-Put-Options but we have implemented other versions of options too
### Weak Scaling :
Weak scaling on the number of MPI nodes, if threads per node = 1

![Alt text](<charts/Nodes vs Speed-up (1 thread_node) weak scaling.png>)

Weak scaling on the number of MPI nodes, if threads per node = 2

![Alt text](<charts/Nodes vs Speed-up (2 Threads_Node) weak_scaling.png>)

Weak scaling on the number of MPI nodes, if threads per node = 4

![Alt text](<charts/Nodes and Speed-up (4 threads_node) weak_scaling.png>)

Weak Scaling on the number of threads (Keeping nodes = 1)

![Alt text](<charts/Speed-up vs Threads (Number of nodes = 1) Weak Scaling.png>)

### Strong Scaling:

Strong scaling on the number of MPI nodes, if threads per node = 1

![Alt text](<charts/Strong_Speedup_Nodes for 1 Thread.png>)

Strong scaling on the number of MPI nodes, if threads per node = 2

![Alt text](<charts/Strong_Speedup_Nodes for 2 Threads.png>)

Strong scaling on the number of MPI nodes, if threads per node = 4

![Alt text](<charts/Strong_Speedup_Nodes for 4 Threads.png>)

Strong scaling on the number of threads, if node = 1

![Alt text](<charts/Strong_Speedup_Threads for 1 Node.png>)

Strong scaling on the number of threads, if node = 2

![Alt text](<charts/Strong_Speedup_Threads for 2 Nodes.png>)

Strong scaling on the number of threads, if node = 4

![Alt text](<charts/Strong_Speedup_Threads for 4 Nodes.png>)

## References

1. "Monte Carlo Simulations In CUDA - Barrier Option Pricing",  QuantStart, [Link](https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/) <br>
