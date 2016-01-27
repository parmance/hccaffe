# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }




# Installation #

## Install Caffe Prerequisites

* BLAS via ATLAS, MKL, or OpenBLAS.
* Boost >= 1.55
* OpenCV >= 2.4 including 3.0
* protobuf, glog, gflags
* IO libraries hdf5, leveldb, snappy, lmdb

reference to http://caffe.berkeleyvision.org/installation.html

## Install HCC
* Download HCC
[hcc binary packages](https://bitbucket.org/multicoreware/hcc/downloads) : Ubuntu x86-64 debian package, or x86-64 .tar.gz tarballs are available.
      

* Install Ubuntu binary packages for x86-64

      By default, HCC would be installed to /opt/hcc.
      To install HCC, download hcc DEB files from links above, and then: 

      
```
#!shell

sudo dpkg -i hcc-<version>-Linux.deb
```



*       You can also choose to download hcc tar.gz files from links above, and then:


```
#!shell

sudo tar zxvf hcc-<vers```ion>-Linux.tar.gz
```



*  Setting up environment variables
      Use the following command to set up environment variables needed for HCC and add it into PA#!shellTH:


```
#!shell

      export HCC_HOME=/opt/hcc
      export PATH=$H```CC_HOME/bin:$PATH
```



   
* You can also install from source code or binary tarballs, please reference to https://bitbucket.org/multicoreware/hcc/wiki/Home

## HCC Support

* Clone Caffe from bitbucket.org.

```
#!shell

git clone https://bitbucket.org/multicoreware/hccaffe.git
```
* Checkout c++amp branch which contain code of support CPP AMP


```
#!shell

#checkout c++amp branch
git checkout -b c++amp origin/c++amp
```
 

* If you want build Caffe with CPP AMP support, please add “USE_CPPAMP” macro to Makefile.config.example file. The make command compiles the CPP AMP code when set USE_CPPAMP := 1 and CPU_ONLY := 0.

```
#!Makefile
# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 0
# C++AMP acceleration switch (uncomment to build with C++AMP).
USE_CPPAMP := 1

```
* The original CUDA code can also work if set USE_CPPAMP := 0 and CPU_ONLY := 0.

```
#!Makefile
# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 0
# C++AMP acceleration switch (uncomment to build with C++AMP).
USE_CPPAMP := 0

```

* Rename Makefile.config.example to Makefile.config


```
#!shell

cp Makefile.config.example Makefile.config
```




## Build Caffe with HCC support

* If you installed the binary package of HCC, then you can build Caffe directly by running 

```
#!shell


make
make test
```

* If you build HCC from source code, please modify Makefile.BuildKalmar file first, you need set you build folder of HCC to CLAMP_PREFIX. And then build AMP Caffe with 

```
#!shell

make -f Makefie.BuildKalmar
make test -f Makefile.BuildKalmar
```


* The last step is verifying the correctness.

```
#!Shell


make all
make test
make runtest
```