# CSCI596_Option-Pricing

## How to run:

#### Single CPU, Single GPU running on multiple Cuda threads and a single CPU thread:
```console
foo@bar:~$ make main
foo@bar:~$ sbatch main.sl
```

#### Multiple CPU, Multiple GPU running on multiple Cuda threads and multiple CPU threads:
```console
foo@bar:~$ make main_omp
foo@bar:~$ sbatch main_omp.sl
```
#### Muliple Node, Multiple CPUs, Multiple GPUs running on multiple Cuda threads and multiple CPU threads
(Not finished yet)
## Brief background

This project simulates barrier options, where the pay-off not only depends on the price of the underlying asset at maturity, but also depends on if the underlying hit a price known as the barrier.

We use a down-and-out kind of call option right now although we are to adding more options to run soon.
Currently, we have also set the rebate price as 0, but we are also working on adding the option to allow a different rebate price.

## Methodology

### Architecture of the system we run this on:
We run our program on the CARC High Performance computing cluster.
The architecture looks somehting like this:

![Alt text](image.png)
<p>
The computation is divided between nodes and each node runs a process.
Each node interacts with each other using the Message Passing Interface (MPI)
<p>
Each node has multiple CPU Cores in it and these cores can run multiple threads for each node process. These threads use OpenMP for interaction and parallalization of threads
<p>
Each node also has accelerated GPU units associated with it where each unit can run multiple CUDA threads.


### How the code works:

We use the formula given in [this](https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/) article by QuantStart, to simulate the underlying price, and the discretized Euler method version comes down to this:
$$ S_t = S_{t-1}\: +\:  \mu S_{t-1} \Delta t\:+ \: \sigma  S_{n-1}\Delta W_t  $$

$S_t$ : The price of the underlying at time t <br>
$\mu$ : The expected return <br>
$\sigma$ : The expected volatility<br>
$\Delta t$: The time difference between each iteration <br>
$\Delta W_t$: Random number drawn from a distribution with mean 0 and variance $ \Delta t $ (Brownian motion component) <br>
<br>

The code generates a an array of random elements $\Delta W_t$ and simulates the price motion according to it. 
We have 3 versions of the code:
<ol>
<li> <p>Simple single, threaded version
<li> <p>Single CPU, Single GPU running on multiple Cuda threads and a single CPU thread
<li><p> Multiple CPU, Multiple GPU running on multiple Cuda threads and multiple CPU threads
<br> Each CPU runs a sinsgle CPU thread. Here we make sure to allocate a GPU to every CPU and reduction of the result from multiple CUDA threads running on Single GPU is performed on a Single CPU thread.  
<li> <p> Muliple Node, Multiple CPUs, Multiple GPUs running on multiple Cuda threads and multiple CPU threads<br>
Each node runs a version of Multiple CPU, Multiple GPU running on multiple Cuda threads and multiple CPU threads
</ol>


## References

<ol>
<li> "Monte Carlo Simulations In CUDA - Barrier Option Pricing",  QuantStart, <a url=https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/>Link</a>