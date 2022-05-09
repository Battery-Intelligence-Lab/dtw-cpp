---
layout: default
title: Goal
nav_order: 2
---

# Dynamic Time Warping

The potential approaches for time series clustering can be broadly defined as using a distance metric on the raw data (distance-based), or extracting features or models from the raw data and then clustering. Distance-based methods have many advantages. The most significant advantage is that using the raw data means the results are not biased as can be the case in methods using inputs extracted from the data, because the features or models extracted have to be chosen prior to the clustering process. However, there are also potential disadvantages. Primarily, an incorrect choice of distance-metric can lead to non-logical clusters and picking a correct distance metric can be a very complex task.

Dynamic time warping (DTW) was chosen as the most appropriate distance metric due to it's ability to handle different length inputs and robustness against time shifts, ensuring usage events don't have to occur at the same timestamp for their similarity to be recognised.


## Speed of calculation & data

The main advantage this code is that it fast and relatively flexible to use. The calculation speed depends on what you are simulating. Below are a few tips.

-  <p style='text-align: justify;'> Calculating a CV takes much longer than a CC. As reference, calculating 100 1C cycles with only CC takes about 0.9 second while calculating 100 1C cycles with also a 
    CV phase on charge (with a current limit of 0.05C) takes 1.9 second. This depends of course on how long the CV phase takes (a lower diffusion constant will give a longer 
    CV and therefore increase calculation time). </p>
- <p style='text-align: justify;'> The code can record periodic values of current, voltage and temperature. This slows down the code and creates a lot of data, especially when you simulate long term degradation. 
    Writing the data to files takes long, with the exact time depending on the speed of your hard disk. As a reference, the same 100 1C cycles (CC only) as before but with a 5s 
    time recording interval takes 18 seconds, with the vast majority of the extra time spent on writing 17 MB of data to the hard disk. Simulating the 100 cycles with an additional CV 
    charge and 5s data collection takes 22 seconds and 21 MB is written. When simulating degradation experiments, gigabytes of data will be generated and it can take up to hours to write all this data 
    (even though the actual calculations take only a couple of minutes). Especially for profile ageing (drive cycles), the amount of data generated (and time to write this data) is huge. 
    But it can be used to compare the simulation with lab data. </p>

- <p style='text-align: justify;'> Some degradation models significantly increase the calculation time. Most models only add about 10% to the calculation time but some models have more impact. E.g. The degradation models with the stress model from Dai (LAM model 1 or CS model 2) double the calculation time. </p>
