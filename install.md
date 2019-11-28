I tried GCC 5.5.0 but it didn't work, but supports GCC 6 at maximum. Therefore I installed
GCC 6.5.0 and set the correct symlink for CUDA:

```sh
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
```

However, this still gave me a lot of warnings so probably the best option is to
install GCC 5.4.0 as recommended in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements).
