# PyTorch itself is currently broken

## symptom 1:
```bash
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$ pip install -e ./VideoMamba/causal-conv1d --no-build-isolation
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Obtaining file:///media/zhangbolin/hu/OSGs/open_source/VideoMamba/causal-conv1d
  Checking if build backend supports build_editable ... done
  Preparing editable metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Preparing editable metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [19 lines of output]
      /media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
        warn(
      Traceback (most recent call last):
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 209, in prepare_metadata_for_build_editable
          return hook(metadata_directory, config_settings)
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/setuptools/build_meta.py", line 484, in prepare_metadata_for_build_editable
          return self.prepare_metadata_for_build_wheel(
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/setuptools/build_meta.py", line 378, in prepare_metadata_for_build_wheel
          self.run_setup()
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 18, in <module>
        File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/torch/__init__.py", line 218, in <module>
          from torch._C import *  # noqa: F403
      ImportError: /media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> from file:///media/zhangbolin/hu/OSGs/open_source/VideoMamba/causal-conv1d

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$ cat ./VideoMamba/causal-conv1d/pyproject.toml 
[build-system]
requires = ["setuptools>=61", "wheel", "torch"]
build-backend = "setuptools.build_meta"
```

## symptom 2:
```bash
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$ python -c "import torch; print(torch.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/torch/__init__.py", line 218, in <module>
    from torch._C import *  # noqa: F403
ImportError: /media/zhangbolin/conda_envs/sdgan/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```
## reason:
The error confirms that PyTorch itself is currently broken in your (sdgan) conda environment: even a simple import torch fails due to the missing symbol iJIT_NotifyEvent in libtorch_cpu.so.
This is a well-known compatibility issue between PyTorch's official conda binaries (built against MKL ~2024.0–2024.2) and newer Intel MKL versions (2024.1+ or especially 2025.x), where Intel removed/changed some symbols from their ITT (Instrumentation and Tracing Technology) API. Conda's dependency solver often pulls in the latest MKL because PyTorch only loosely requires mkl >= 2018.


## fix:
```bash
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$ conda install "mkl<2025" -y
Channels:
 - pytorch
 - defaults
 - nvidia
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /media/zhangbolin/conda_envs/sdgan

  added / updated specs:
    - mkl[version='<2025']


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    mkl-service-2.4.0          |  py310h5eee18b_2          68 KB
    mkl_fft-1.3.11             |  py310h5eee18b_0         198 KB
    mkl_random-1.2.8           |  py310h1128e8f_0         318 KB
    numpy-2.2.5                |  py310h2470af2_0          13 KB
    numpy-base-2.2.5           |  py310h06ae042_0         8.0 MB
    setuptools-72.1.0          |  py310h06a4308_0         2.4 MB
    tbb-devel-2021.8.0         |       hdb19cb5_0         1.1 MB
    ------------------------------------------------------------
                                           Total:        12.1 MB

The following packages will be DOWNGRADED:

  intel-openmp                       2025.0.0-h06a4308_1171 --> 2023.1.0-hdb19cb5_46306 
  mkl                                 2025.0.0-hacee8c2_941 --> 2023.1.0-h213fc3f_46344 
  mkl-service                         2.5.2-py310hacdc0fc_0 --> 2.4.0-py310h5eee18b_2 
  mkl_fft                             2.1.1-py310h8fe796d_0 --> 1.3.11-py310h5eee18b_0 
  mkl_random                          1.3.0-py310h505adc9_0 --> 1.2.8-py310h1128e8f_0 
  numpy                               2.2.5-py310h64c44e4_2 --> 2.2.5-py310h2470af2_0 
  numpy-base                          2.2.5-py310he1678cf_2 --> 2.2.5-py310h06ae042_0 
  setuptools                        80.10.2-py310h06a4308_0 --> 72.1.0-py310h06a4308_0 
  tbb                                   2022.3.0-h698db13_0 --> 2021.8.0-hdb19cb5_0 
  tbb-devel                             2022.3.0-h698db13_0 --> 2021.8.0-hdb19cb5_0 



Downloading and Extracting Packages:
                                                                                                                               
Preparing transaction: done                                                                                                    
Verifying transaction: done                                                                                                    
Executing transaction: done                                                                                                    
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$ python -c "import torch; print(torch.__version__)"          
1.13.1                                                                                                                         
(sdgan)zhangbolin@nbu-cs-02:/media/zhangbolin/hu/OSGs/open_source$                                                          
```
