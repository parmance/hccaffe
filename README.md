# ** HCC backend Implementation for Caffe ** #

##Introduction: ##

This repository hosts the HCC backend implementation project for  [Caffe](https://github.com/BVLC/caffe). To know what HCC is please refer [here](https://bitbucket.org/multicoreware/hcc/wiki/Home). Caffe framework currently has a CUDA backend support targeting NVidia devices.  The goal of this project is to develop  HCC counterparts targeting modern AMD devices. This project mainly targets the linux platform and makes use of the linux-based HCC compiler implementation hosted [here](https://bitbucket.org/multicoreware/hcc/wiki/Home). 

##Prerequisites: ##

**Hardware Requirements:**

* CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 
* System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
* Hard Drive > 200GB (Better if SSD or NVMe driver  for NN application over multiple GPUs)
* Minimum GPU Memory (Global) > 2GB

**GPU SDK and driver Requirements:**

* dGPUs: AMD R9 Fury X, R9 Fury, R9 Nano
* APUs: AMD APU Kaveri or Carrizo

**System software requirements:**

* Ubuntu 14.04 trusty
* GCC 4.6 and later
* CPP 4.6 and later (come with GCC package)
* python 2.7 and later
* HCC 0.9 from [here](https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb)


**Tools and Misc Requirements:**

* git 1.9 and later
* cmake 2.6 and later (2.6 and 2.8 are tested)



**Ubuntu Packages requirements:**

* libc6-dev-i386
* liblapack-dev
* graphicsmagick
* libboost-all-dev
* lua5.1


## Tested Environment so far: 

This section enumerates the list of tested combinations of Hardware and system software

**GPU Cards tested:**

* Radeon R9 Nano
* Radeon R9 FuryX 
* Radeon R9 Fury 
* Kaveri and Carizo APU

**Driver versions tested**  

* Boltzmann Early Release Driver for dGPU

   ROCM 1.0 Release : https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md
     
* Traditional HSA driver for APU (Kaveri)

**Desktop System Tested**

* Supermicro SYS-7048GR-TR  Tower 4 R9 Nano
* ASUS X99-E WS motherboard with 4 R9 Nano
* Gigabyte GA-X79S 2 AMD R9 Nano

**Server System Tested**

* Supermicro SYS 2028GR-THT  6 R9 NANO
* Supermicro SYS-1028GQ-TRT 4 R9 NANO
* Supermicro SYS-7048GR-TR Tower 4 R9 NANO
 

## Installation Flow: 

A. ROCM 1.0 Installation (If not done so far)

B. HCCaffe Build

C. Unit Testing


## Installation Steps in detail:

### A. ROCM 1.0 Installation: 

  To Know more about ROCM  refer https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

  a. Installing Debian ROCM repositories
     
  Before proceeding, make sure to completely uninstall any pre-release ROCm packages
     
  Refer https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages for instructions to remove pre-release ROCM packages
     
  Steps to install rocm package are 
     
      * wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
      
      * sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ trusty main > /etc/apt/sources.list.d/rocm.list'
     
      * sudo apt-get update
      
      * sudo apt-get install rocm
      
      * Reboot the system
      
  b. Once Reboot, verify the installation
    
  To verify that the ROCm stack completed successfully you can execute to HSA vector_copy sample application:

       * cd /opt/rocm/hsa/sample
        
       * make
       
       * ./vector_copy


      
### B. Hccaffe Build Steps:

HcCaffe currently could be built in one of the following two ways

i) Using prebuilt hcc-hsail compiler under /opt/rocm/hcc-hsail
        
                      (or)
ii) Using hcc-hsail built from source


**(i) Using Prebuild hcc-hsail**:

    (Assumption /opt/rocm/hcc-hsail hosts the compiler)
   
    * make

    * make test

**(ii) Using HCC built from source*:

   * export MCWHCCBUILD=<path to HCC build>

   * make -f Makefile.BuildHCC

   * make test -f Makefile.BuildHCC




## C. Unit Testing ##

After done with A and B, Now its time to test. Run the following commands to perform unit testing of different components of Caffe.

             ./build/test/test_all.testbin