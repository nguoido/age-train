2021-06-12 15:14:33.632347: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2021-06-12 15:14:35.714378: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2021-06-12 15:14:35.745795: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.746542: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
2021-06-12 15:14:35.746600: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2021-06-12 15:14:35.749215: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11
2021-06-12 15:14:35.749298: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11
2021-06-12 15:14:35.751072: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcufft.so.10
2021-06-12 15:14:35.751440: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcurand.so.10
2021-06-12 15:14:35.753427: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusolver.so.10
2021-06-12 15:14:35.754017: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusparse.so.11
2021-06-12 15:14:35.754237: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudnn.so.8
2021-06-12 15:14:35.754352: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.755155: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.755867: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-06-12 15:14:35.756487: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.757360: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
2021-06-12 15:14:35.757460: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.758306: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:35.759004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2021-06-12 15:14:35.759074: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2021-06-12 15:14:36.238242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-06-12 15:14:36.238303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0 
2021-06-12 15:14:36.238328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N 
2021-06-12 15:14:36.238581: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:36.239483: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:36.240280: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-06-12 15:14:36.241109: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
2021-06-12 15:14:36.241176: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10818 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
WARNING:tensorflow:Collective ops is not configured at program startup. Some performance features may not be enabled.
[2021-06-12 15:14:36,243][tensorflow][WARNING] - Collective ops is not configured at program startup. Some performance features may not be enabled.
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
[2021-06-12 15:14:36,247][tensorflow][INFO] - Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,299][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,301][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,304][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,305][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,313][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,318][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,350][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,351][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,353][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
[2021-06-12 15:14:36,354][tensorflow][INFO] - Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5
43941888/43941136 [==============================] - 0s 0us/step
>>>>>>>>>>>>>>>>>> START model.summary() >>>>>>>>>>>>>>>>>>
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 224, 224, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
normalization (Normalization)   (None, 224, 224, 3)  7           rescaling[0][0]                  
__________________________________________________________________________________________________
stem_conv_pad (ZeroPadding2D)   (None, 225, 225, 3)  0           normalization[0][0]              
__________________________________________________________________________________________________
stem_conv (Conv2D)              (None, 112, 112, 40) 1080        stem_conv_pad[0][0]              
__________________________________________________________________________________________________
stem_bn (BatchNormalization)    (None, 112, 112, 40) 160         stem_conv[0][0]                  
__________________________________________________________________________________________________
stem_activation (Activation)    (None, 112, 112, 40) 0           stem_bn[0][0]                    
__________________________________________________________________________________________________
block1a_dwconv (DepthwiseConv2D (None, 112, 112, 40) 360         stem_activation[0][0]            
__________________________________________________________________________________________________
block1a_bn (BatchNormalization) (None, 112, 112, 40) 160         block1a_dwconv[0][0]             
__________________________________________________________________________________________________
block1a_activation (Activation) (None, 112, 112, 40) 0           block1a_bn[0][0]                 
__________________________________________________________________________________________________
block1a_se_squeeze (GlobalAvera (None, 40)           0           block1a_activation[0][0]         
__________________________________________________________________________________________________
block1a_se_reshape (Reshape)    (None, 1, 1, 40)     0           block1a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1a_se_reduce (Conv2D)      (None, 1, 1, 10)     410         block1a_se_reshape[0][0]         
__________________________________________________________________________________________________
block1a_se_expand (Conv2D)      (None, 1, 1, 40)     440         block1a_se_reduce[0][0]          
__________________________________________________________________________________________________
block1a_se_excite (Multiply)    (None, 112, 112, 40) 0           block1a_activation[0][0]         
                                                                 block1a_se_expand[0][0]          
__________________________________________________________________________________________________
block1a_project_conv (Conv2D)   (None, 112, 112, 24) 960         block1a_se_excite[0][0]          
__________________________________________________________________________________________________
block1a_project_bn (BatchNormal (None, 112, 112, 24) 96          block1a_project_conv[0][0]       
__________________________________________________________________________________________________
block1b_dwconv (DepthwiseConv2D (None, 112, 112, 24) 216         block1a_project_bn[0][0]         
__________________________________________________________________________________________________
block1b_bn (BatchNormalization) (None, 112, 112, 24) 96          block1b_dwconv[0][0]             
__________________________________________________________________________________________________
block1b_activation (Activation) (None, 112, 112, 24) 0           block1b_bn[0][0]                 
__________________________________________________________________________________________________
block1b_se_squeeze (GlobalAvera (None, 24)           0           block1b_activation[0][0]         
__________________________________________________________________________________________________
block1b_se_reshape (Reshape)    (None, 1, 1, 24)     0           block1b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1b_se_reduce (Conv2D)      (None, 1, 1, 6)      150         block1b_se_reshape[0][0]         
__________________________________________________________________________________________________
block1b_se_expand (Conv2D)      (None, 1, 1, 24)     168         block1b_se_reduce[0][0]          
__________________________________________________________________________________________________
block1b_se_excite (Multiply)    (None, 112, 112, 24) 0           block1b_activation[0][0]         
                                                                 block1b_se_expand[0][0]          
__________________________________________________________________________________________________
block1b_project_conv (Conv2D)   (None, 112, 112, 24) 576         block1b_se_excite[0][0]          
__________________________________________________________________________________________________
block1b_project_bn (BatchNormal (None, 112, 112, 24) 96          block1b_project_conv[0][0]       
__________________________________________________________________________________________________
block1b_drop (Dropout)          (None, 112, 112, 24) 0           block1b_project_bn[0][0]         
__________________________________________________________________________________________________
block1b_add (Add)               (None, 112, 112, 24) 0           block1b_drop[0][0]               
                                                                 block1a_project_bn[0][0]         
__________________________________________________________________________________________________
block2a_expand_conv (Conv2D)    (None, 112, 112, 144 3456        block1b_add[0][0]                
__________________________________________________________________________________________________
block2a_expand_bn (BatchNormali (None, 112, 112, 144 576         block2a_expand_conv[0][0]        
__________________________________________________________________________________________________
block2a_expand_activation (Acti (None, 112, 112, 144 0           block2a_expand_bn[0][0]          
__________________________________________________________________________________________________
block2a_dwconv_pad (ZeroPadding (None, 113, 113, 144 0           block2a_expand_activation[0][0]  
__________________________________________________________________________________________________
block2a_dwconv (DepthwiseConv2D (None, 56, 56, 144)  1296        block2a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block2a_bn (BatchNormalization) (None, 56, 56, 144)  576         block2a_dwconv[0][0]             
__________________________________________________________________________________________________
block2a_activation (Activation) (None, 56, 56, 144)  0           block2a_bn[0][0]                 
__________________________________________________________________________________________________
block2a_se_squeeze (GlobalAvera (None, 144)          0           block2a_activation[0][0]         
__________________________________________________________________________________________________
block2a_se_reshape (Reshape)    (None, 1, 1, 144)    0           block2a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block2a_se_reduce (Conv2D)      (None, 1, 1, 6)      870         block2a_se_reshape[0][0]         
__________________________________________________________________________________________________
block2a_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block2a_se_reduce[0][0]          
__________________________________________________________________________________________________
block2a_se_excite (Multiply)    (None, 56, 56, 144)  0           block2a_activation[0][0]         
                                                                 block2a_se_expand[0][0]          
__________________________________________________________________________________________________
block2a_project_conv (Conv2D)   (None, 56, 56, 32)   4608        block2a_se_excite[0][0]          
__________________________________________________________________________________________________
block2a_project_bn (BatchNormal (None, 56, 56, 32)   128         block2a_project_conv[0][0]       
__________________________________________________________________________________________________
block2b_expand_conv (Conv2D)    (None, 56, 56, 192)  6144        block2a_project_bn[0][0]         
__________________________________________________________________________________________________
block2b_expand_bn (BatchNormali (None, 56, 56, 192)  768         block2b_expand_conv[0][0]        
__________________________________________________________________________________________________
block2b_expand_activation (Acti (None, 56, 56, 192)  0           block2b_expand_bn[0][0]          
__________________________________________________________________________________________________
block2b_dwconv (DepthwiseConv2D (None, 56, 56, 192)  1728        block2b_expand_activation[0][0]  
__________________________________________________________________________________________________
block2b_bn (BatchNormalization) (None, 56, 56, 192)  768         block2b_dwconv[0][0]             
__________________________________________________________________________________________________
block2b_activation (Activation) (None, 56, 56, 192)  0           block2b_bn[0][0]                 
__________________________________________________________________________________________________
block2b_se_squeeze (GlobalAvera (None, 192)          0           block2b_activation[0][0]         
__________________________________________________________________________________________________
block2b_se_reshape (Reshape)    (None, 1, 1, 192)    0           block2b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block2b_se_reduce (Conv2D)      (None, 1, 1, 8)      1544        block2b_se_reshape[0][0]         
__________________________________________________________________________________________________
block2b_se_expand (Conv2D)      (None, 1, 1, 192)    1728        block2b_se_reduce[0][0]          
__________________________________________________________________________________________________
block2b_se_excite (Multiply)    (None, 56, 56, 192)  0           block2b_activation[0][0]         
                                                                 block2b_se_expand[0][0]          
__________________________________________________________________________________________________
block2b_project_conv (Conv2D)   (None, 56, 56, 32)   6144        block2b_se_excite[0][0]          
__________________________________________________________________________________________________
block2b_project_bn (BatchNormal (None, 56, 56, 32)   128         block2b_project_conv[0][0]       
__________________________________________________________________________________________________
block2b_drop (Dropout)          (None, 56, 56, 32)   0           block2b_project_bn[0][0]         
__________________________________________________________________________________________________
block2b_add (Add)               (None, 56, 56, 32)   0           block2b_drop[0][0]               
                                                                 block2a_project_bn[0][0]         
__________________________________________________________________________________________________
block2c_expand_conv (Conv2D)    (None, 56, 56, 192)  6144        block2b_add[0][0]                
__________________________________________________________________________________________________
block2c_expand_bn (BatchNormali (None, 56, 56, 192)  768         block2c_expand_conv[0][0]        
__________________________________________________________________________________________________
block2c_expand_activation (Acti (None, 56, 56, 192)  0           block2c_expand_bn[0][0]          
__________________________________________________________________________________________________
block2c_dwconv (DepthwiseConv2D (None, 56, 56, 192)  1728        block2c_expand_activation[0][0]  
__________________________________________________________________________________________________
block2c_bn (BatchNormalization) (None, 56, 56, 192)  768         block2c_dwconv[0][0]             
__________________________________________________________________________________________________
block2c_activation (Activation) (None, 56, 56, 192)  0           block2c_bn[0][0]                 
__________________________________________________________________________________________________
block2c_se_squeeze (GlobalAvera (None, 192)          0           block2c_activation[0][0]         
__________________________________________________________________________________________________
block2c_se_reshape (Reshape)    (None, 1, 1, 192)    0           block2c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block2c_se_reduce (Conv2D)      (None, 1, 1, 8)      1544        block2c_se_reshape[0][0]         
__________________________________________________________________________________________________
block2c_se_expand (Conv2D)      (None, 1, 1, 192)    1728        block2c_se_reduce[0][0]          
__________________________________________________________________________________________________
block2c_se_excite (Multiply)    (None, 56, 56, 192)  0           block2c_activation[0][0]         
                                                                 block2c_se_expand[0][0]          
__________________________________________________________________________________________________
block2c_project_conv (Conv2D)   (None, 56, 56, 32)   6144        block2c_se_excite[0][0]          
__________________________________________________________________________________________________
block2c_project_bn (BatchNormal (None, 56, 56, 32)   128         block2c_project_conv[0][0]       
__________________________________________________________________________________________________
block2c_drop (Dropout)          (None, 56, 56, 32)   0           block2c_project_bn[0][0]         
__________________________________________________________________________________________________
block2c_add (Add)               (None, 56, 56, 32)   0           block2c_drop[0][0]               
                                                                 block2b_add[0][0]                
__________________________________________________________________________________________________
block3a_expand_conv (Conv2D)    (None, 56, 56, 192)  6144        block2c_add[0][0]                
__________________________________________________________________________________________________
block3a_expand_bn (BatchNormali (None, 56, 56, 192)  768         block3a_expand_conv[0][0]        
__________________________________________________________________________________________________
block3a_expand_activation (Acti (None, 56, 56, 192)  0           block3a_expand_bn[0][0]          
__________________________________________________________________________________________________
block3a_dwconv_pad (ZeroPadding (None, 59, 59, 192)  0           block3a_expand_activation[0][0]  
__________________________________________________________________________________________________
block3a_dwconv (DepthwiseConv2D (None, 28, 28, 192)  4800        block3a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block3a_bn (BatchNormalization) (None, 28, 28, 192)  768         block3a_dwconv[0][0]             
__________________________________________________________________________________________________
block3a_activation (Activation) (None, 28, 28, 192)  0           block3a_bn[0][0]                 
__________________________________________________________________________________________________
block3a_se_squeeze (GlobalAvera (None, 192)          0           block3a_activation[0][0]         
__________________________________________________________________________________________________
block3a_se_reshape (Reshape)    (None, 1, 1, 192)    0           block3a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block3a_se_reduce (Conv2D)      (None, 1, 1, 8)      1544        block3a_se_reshape[0][0]         
__________________________________________________________________________________________________
block3a_se_expand (Conv2D)      (None, 1, 1, 192)    1728        block3a_se_reduce[0][0]          
__________________________________________________________________________________________________
block3a_se_excite (Multiply)    (None, 28, 28, 192)  0           block3a_activation[0][0]         
                                                                 block3a_se_expand[0][0]          
__________________________________________________________________________________________________
block3a_project_conv (Conv2D)   (None, 28, 28, 48)   9216        block3a_se_excite[0][0]          
__________________________________________________________________________________________________
block3a_project_bn (BatchNormal (None, 28, 28, 48)   192         block3a_project_conv[0][0]       
__________________________________________________________________________________________________
block3b_expand_conv (Conv2D)    (None, 28, 28, 288)  13824       block3a_project_bn[0][0]         
__________________________________________________________________________________________________
block3b_expand_bn (BatchNormali (None, 28, 28, 288)  1152        block3b_expand_conv[0][0]        
__________________________________________________________________________________________________
block3b_expand_activation (Acti (None, 28, 28, 288)  0           block3b_expand_bn[0][0]          
__________________________________________________________________________________________________
block3b_dwconv (DepthwiseConv2D (None, 28, 28, 288)  7200        block3b_expand_activation[0][0]  
__________________________________________________________________________________________________
block3b_bn (BatchNormalization) (None, 28, 28, 288)  1152        block3b_dwconv[0][0]             
__________________________________________________________________________________________________
block3b_activation (Activation) (None, 28, 28, 288)  0           block3b_bn[0][0]                 
__________________________________________________________________________________________________
block3b_se_squeeze (GlobalAvera (None, 288)          0           block3b_activation[0][0]         
__________________________________________________________________________________________________
block3b_se_reshape (Reshape)    (None, 1, 1, 288)    0           block3b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block3b_se_reduce (Conv2D)      (None, 1, 1, 12)     3468        block3b_se_reshape[0][0]         
__________________________________________________________________________________________________
block3b_se_expand (Conv2D)      (None, 1, 1, 288)    3744        block3b_se_reduce[0][0]          
__________________________________________________________________________________________________
block3b_se_excite (Multiply)    (None, 28, 28, 288)  0           block3b_activation[0][0]         
                                                                 block3b_se_expand[0][0]          
__________________________________________________________________________________________________
block3b_project_conv (Conv2D)   (None, 28, 28, 48)   13824       block3b_se_excite[0][0]          
__________________________________________________________________________________________________
block3b_project_bn (BatchNormal (None, 28, 28, 48)   192         block3b_project_conv[0][0]       
__________________________________________________________________________________________________
block3b_drop (Dropout)          (None, 28, 28, 48)   0           block3b_project_bn[0][0]         
__________________________________________________________________________________________________
block3b_add (Add)               (None, 28, 28, 48)   0           block3b_drop[0][0]               
                                                                 block3a_project_bn[0][0]         
__________________________________________________________________________________________________
block3c_expand_conv (Conv2D)    (None, 28, 28, 288)  13824       block3b_add[0][0]                
__________________________________________________________________________________________________
block3c_expand_bn (BatchNormali (None, 28, 28, 288)  1152        block3c_expand_conv[0][0]        
__________________________________________________________________________________________________
block3c_expand_activation (Acti (None, 28, 28, 288)  0           block3c_expand_bn[0][0]          
__________________________________________________________________________________________________
block3c_dwconv (DepthwiseConv2D (None, 28, 28, 288)  7200        block3c_expand_activation[0][0]  
__________________________________________________________________________________________________
block3c_bn (BatchNormalization) (None, 28, 28, 288)  1152        block3c_dwconv[0][0]             
__________________________________________________________________________________________________
block3c_activation (Activation) (None, 28, 28, 288)  0           block3c_bn[0][0]                 
__________________________________________________________________________________________________
block3c_se_squeeze (GlobalAvera (None, 288)          0           block3c_activation[0][0]         
__________________________________________________________________________________________________
block3c_se_reshape (Reshape)    (None, 1, 1, 288)    0           block3c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block3c_se_reduce (Conv2D)      (None, 1, 1, 12)     3468        block3c_se_reshape[0][0]         
__________________________________________________________________________________________________
block3c_se_expand (Conv2D)      (None, 1, 1, 288)    3744        block3c_se_reduce[0][0]          
__________________________________________________________________________________________________
block3c_se_excite (Multiply)    (None, 28, 28, 288)  0           block3c_activation[0][0]         
                                                                 block3c_se_expand[0][0]          
__________________________________________________________________________________________________
block3c_project_conv (Conv2D)   (None, 28, 28, 48)   13824       block3c_se_excite[0][0]          
__________________________________________________________________________________________________
block3c_project_bn (BatchNormal (None, 28, 28, 48)   192         block3c_project_conv[0][0]       
__________________________________________________________________________________________________
block3c_drop (Dropout)          (None, 28, 28, 48)   0           block3c_project_bn[0][0]         
__________________________________________________________________________________________________
block3c_add (Add)               (None, 28, 28, 48)   0           block3c_drop[0][0]               
                                                                 block3b_add[0][0]                
__________________________________________________________________________________________________
block4a_expand_conv (Conv2D)    (None, 28, 28, 288)  13824       block3c_add[0][0]                
__________________________________________________________________________________________________
block4a_expand_bn (BatchNormali (None, 28, 28, 288)  1152        block4a_expand_conv[0][0]        
__________________________________________________________________________________________________
block4a_expand_activation (Acti (None, 28, 28, 288)  0           block4a_expand_bn[0][0]          
__________________________________________________________________________________________________
block4a_dwconv_pad (ZeroPadding (None, 29, 29, 288)  0           block4a_expand_activation[0][0]  
__________________________________________________________________________________________________
block4a_dwconv (DepthwiseConv2D (None, 14, 14, 288)  2592        block4a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block4a_bn (BatchNormalization) (None, 14, 14, 288)  1152        block4a_dwconv[0][0]             
__________________________________________________________________________________________________
block4a_activation (Activation) (None, 14, 14, 288)  0           block4a_bn[0][0]                 
__________________________________________________________________________________________________
block4a_se_squeeze (GlobalAvera (None, 288)          0           block4a_activation[0][0]         
__________________________________________________________________________________________________
block4a_se_reshape (Reshape)    (None, 1, 1, 288)    0           block4a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4a_se_reduce (Conv2D)      (None, 1, 1, 12)     3468        block4a_se_reshape[0][0]         
__________________________________________________________________________________________________
block4a_se_expand (Conv2D)      (None, 1, 1, 288)    3744        block4a_se_reduce[0][0]          
__________________________________________________________________________________________________
block4a_se_excite (Multiply)    (None, 14, 14, 288)  0           block4a_activation[0][0]         
                                                                 block4a_se_expand[0][0]          
__________________________________________________________________________________________________
block4a_project_conv (Conv2D)   (None, 14, 14, 96)   27648       block4a_se_excite[0][0]          
__________________________________________________________________________________________________
block4a_project_bn (BatchNormal (None, 14, 14, 96)   384         block4a_project_conv[0][0]       
__________________________________________________________________________________________________
block4b_expand_conv (Conv2D)    (None, 14, 14, 576)  55296       block4a_project_bn[0][0]         
__________________________________________________________________________________________________
block4b_expand_bn (BatchNormali (None, 14, 14, 576)  2304        block4b_expand_conv[0][0]        
__________________________________________________________________________________________________
block4b_expand_activation (Acti (None, 14, 14, 576)  0           block4b_expand_bn[0][0]          
__________________________________________________________________________________________________
block4b_dwconv (DepthwiseConv2D (None, 14, 14, 576)  5184        block4b_expand_activation[0][0]  
__________________________________________________________________________________________________
block4b_bn (BatchNormalization) (None, 14, 14, 576)  2304        block4b_dwconv[0][0]             
__________________________________________________________________________________________________
block4b_activation (Activation) (None, 14, 14, 576)  0           block4b_bn[0][0]                 
__________________________________________________________________________________________________
block4b_se_squeeze (GlobalAvera (None, 576)          0           block4b_activation[0][0]         
__________________________________________________________________________________________________
block4b_se_reshape (Reshape)    (None, 1, 1, 576)    0           block4b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4b_se_reduce (Conv2D)      (None, 1, 1, 24)     13848       block4b_se_reshape[0][0]         
__________________________________________________________________________________________________
block4b_se_expand (Conv2D)      (None, 1, 1, 576)    14400       block4b_se_reduce[0][0]          
__________________________________________________________________________________________________
block4b_se_excite (Multiply)    (None, 14, 14, 576)  0           block4b_activation[0][0]         
                                                                 block4b_se_expand[0][0]          
__________________________________________________________________________________________________
block4b_project_conv (Conv2D)   (None, 14, 14, 96)   55296       block4b_se_excite[0][0]          
__________________________________________________________________________________________________
block4b_project_bn (BatchNormal (None, 14, 14, 96)   384         block4b_project_conv[0][0]       
__________________________________________________________________________________________________
block4b_drop (Dropout)          (None, 14, 14, 96)   0           block4b_project_bn[0][0]         
__________________________________________________________________________________________________
block4b_add (Add)               (None, 14, 14, 96)   0           block4b_drop[0][0]               
                                                                 block4a_project_bn[0][0]         
__________________________________________________________________________________________________
block4c_expand_conv (Conv2D)    (None, 14, 14, 576)  55296       block4b_add[0][0]                
__________________________________________________________________________________________________
block4c_expand_bn (BatchNormali (None, 14, 14, 576)  2304        block4c_expand_conv[0][0]        
__________________________________________________________________________________________________
block4c_expand_activation (Acti (None, 14, 14, 576)  0           block4c_expand_bn[0][0]          
__________________________________________________________________________________________________
block4c_dwconv (DepthwiseConv2D (None, 14, 14, 576)  5184        block4c_expand_activation[0][0]  
__________________________________________________________________________________________________
block4c_bn (BatchNormalization) (None, 14, 14, 576)  2304        block4c_dwconv[0][0]             
__________________________________________________________________________________________________
block4c_activation (Activation) (None, 14, 14, 576)  0           block4c_bn[0][0]                 
__________________________________________________________________________________________________
block4c_se_squeeze (GlobalAvera (None, 576)          0           block4c_activation[0][0]         
__________________________________________________________________________________________________
block4c_se_reshape (Reshape)    (None, 1, 1, 576)    0           block4c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4c_se_reduce (Conv2D)      (None, 1, 1, 24)     13848       block4c_se_reshape[0][0]         
__________________________________________________________________________________________________
block4c_se_expand (Conv2D)      (None, 1, 1, 576)    14400       block4c_se_reduce[0][0]          
__________________________________________________________________________________________________
block4c_se_excite (Multiply)    (None, 14, 14, 576)  0           block4c_activation[0][0]         
                                                                 block4c_se_expand[0][0]          
__________________________________________________________________________________________________
block4c_project_conv (Conv2D)   (None, 14, 14, 96)   55296       block4c_se_excite[0][0]          
__________________________________________________________________________________________________
block4c_project_bn (BatchNormal (None, 14, 14, 96)   384         block4c_project_conv[0][0]       
__________________________________________________________________________________________________
block4c_drop (Dropout)          (None, 14, 14, 96)   0           block4c_project_bn[0][0]         
__________________________________________________________________________________________________
block4c_add (Add)               (None, 14, 14, 96)   0           block4c_drop[0][0]               
                                                                 block4b_add[0][0]                
__________________________________________________________________________________________________
block4d_expand_conv (Conv2D)    (None, 14, 14, 576)  55296       block4c_add[0][0]                
__________________________________________________________________________________________________
block4d_expand_bn (BatchNormali (None, 14, 14, 576)  2304        block4d_expand_conv[0][0]        
__________________________________________________________________________________________________
block4d_expand_activation (Acti (None, 14, 14, 576)  0           block4d_expand_bn[0][0]          
__________________________________________________________________________________________________
block4d_dwconv (DepthwiseConv2D (None, 14, 14, 576)  5184        block4d_expand_activation[0][0]  
__________________________________________________________________________________________________
block4d_bn (BatchNormalization) (None, 14, 14, 576)  2304        block4d_dwconv[0][0]             
__________________________________________________________________________________________________
block4d_activation (Activation) (None, 14, 14, 576)  0           block4d_bn[0][0]                 
__________________________________________________________________________________________________
block4d_se_squeeze (GlobalAvera (None, 576)          0           block4d_activation[0][0]         
__________________________________________________________________________________________________
block4d_se_reshape (Reshape)    (None, 1, 1, 576)    0           block4d_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4d_se_reduce (Conv2D)      (None, 1, 1, 24)     13848       block4d_se_reshape[0][0]         
__________________________________________________________________________________________________
block4d_se_expand (Conv2D)      (None, 1, 1, 576)    14400       block4d_se_reduce[0][0]          
__________________________________________________________________________________________________
block4d_se_excite (Multiply)    (None, 14, 14, 576)  0           block4d_activation[0][0]         
                                                                 block4d_se_expand[0][0]          
__________________________________________________________________________________________________
block4d_project_conv (Conv2D)   (None, 14, 14, 96)   55296       block4d_se_excite[0][0]          
__________________________________________________________________________________________________
block4d_project_bn (BatchNormal (None, 14, 14, 96)   384         block4d_project_conv[0][0]       
__________________________________________________________________________________________________
block4d_drop (Dropout)          (None, 14, 14, 96)   0           block4d_project_bn[0][0]         
__________________________________________________________________________________________________
block4d_add (Add)               (None, 14, 14, 96)   0           block4d_drop[0][0]               
                                                                 block4c_add[0][0]                
__________________________________________________________________________________________________
block4e_expand_conv (Conv2D)    (None, 14, 14, 576)  55296       block4d_add[0][0]                
__________________________________________________________________________________________________
block4e_expand_bn (BatchNormali (None, 14, 14, 576)  2304        block4e_expand_conv[0][0]        
__________________________________________________________________________________________________
block4e_expand_activation (Acti (None, 14, 14, 576)  0           block4e_expand_bn[0][0]          
__________________________________________________________________________________________________
block4e_dwconv (DepthwiseConv2D (None, 14, 14, 576)  5184        block4e_expand_activation[0][0]  
__________________________________________________________________________________________________
block4e_bn (BatchNormalization) (None, 14, 14, 576)  2304        block4e_dwconv[0][0]             
__________________________________________________________________________________________________
block4e_activation (Activation) (None, 14, 14, 576)  0           block4e_bn[0][0]                 
__________________________________________________________________________________________________
block4e_se_squeeze (GlobalAvera (None, 576)          0           block4e_activation[0][0]         
__________________________________________________________________________________________________
block4e_se_reshape (Reshape)    (None, 1, 1, 576)    0           block4e_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4e_se_reduce (Conv2D)      (None, 1, 1, 24)     13848       block4e_se_reshape[0][0]         
__________________________________________________________________________________________________
block4e_se_expand (Conv2D)      (None, 1, 1, 576)    14400       block4e_se_reduce[0][0]          
__________________________________________________________________________________________________
block4e_se_excite (Multiply)    (None, 14, 14, 576)  0           block4e_activation[0][0]         
                                                                 block4e_se_expand[0][0]          
__________________________________________________________________________________________________
block4e_project_conv (Conv2D)   (None, 14, 14, 96)   55296       block4e_se_excite[0][0]          
__________________________________________________________________________________________________
block4e_project_bn (BatchNormal (None, 14, 14, 96)   384         block4e_project_conv[0][0]       
__________________________________________________________________________________________________
block4e_drop (Dropout)          (None, 14, 14, 96)   0           block4e_project_bn[0][0]         
__________________________________________________________________________________________________
block4e_add (Add)               (None, 14, 14, 96)   0           block4e_drop[0][0]               
                                                                 block4d_add[0][0]                
__________________________________________________________________________________________________
block5a_expand_conv (Conv2D)    (None, 14, 14, 576)  55296       block4e_add[0][0]                
__________________________________________________________________________________________________
block5a_expand_bn (BatchNormali (None, 14, 14, 576)  2304        block5a_expand_conv[0][0]        
__________________________________________________________________________________________________
block5a_expand_activation (Acti (None, 14, 14, 576)  0           block5a_expand_bn[0][0]          
__________________________________________________________________________________________________
block5a_dwconv (DepthwiseConv2D (None, 14, 14, 576)  14400       block5a_expand_activation[0][0]  
__________________________________________________________________________________________________
block5a_bn (BatchNormalization) (None, 14, 14, 576)  2304        block5a_dwconv[0][0]             
__________________________________________________________________________________________________
block5a_activation (Activation) (None, 14, 14, 576)  0           block5a_bn[0][0]                 
__________________________________________________________________________________________________
block5a_se_squeeze (GlobalAvera (None, 576)          0           block5a_activation[0][0]         
__________________________________________________________________________________________________
block5a_se_reshape (Reshape)    (None, 1, 1, 576)    0           block5a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5a_se_reduce (Conv2D)      (None, 1, 1, 24)     13848       block5a_se_reshape[0][0]         
__________________________________________________________________________________________________
block5a_se_expand (Conv2D)      (None, 1, 1, 576)    14400       block5a_se_reduce[0][0]          
__________________________________________________________________________________________________
block5a_se_excite (Multiply)    (None, 14, 14, 576)  0           block5a_activation[0][0]         
                                                                 block5a_se_expand[0][0]          
__________________________________________________________________________________________________
block5a_project_conv (Conv2D)   (None, 14, 14, 136)  78336       block5a_se_excite[0][0]          
__________________________________________________________________________________________________
block5a_project_bn (BatchNormal (None, 14, 14, 136)  544         block5a_project_conv[0][0]       
__________________________________________________________________________________________________
block5b_expand_conv (Conv2D)    (None, 14, 14, 816)  110976      block5a_project_bn[0][0]         
__________________________________________________________________________________________________
block5b_expand_bn (BatchNormali (None, 14, 14, 816)  3264        block5b_expand_conv[0][0]        
__________________________________________________________________________________________________
block5b_expand_activation (Acti (None, 14, 14, 816)  0           block5b_expand_bn[0][0]          
__________________________________________________________________________________________________
block5b_dwconv (DepthwiseConv2D (None, 14, 14, 816)  20400       block5b_expand_activation[0][0]  
__________________________________________________________________________________________________
block5b_bn (BatchNormalization) (None, 14, 14, 816)  3264        block5b_dwconv[0][0]             
__________________________________________________________________________________________________
block5b_activation (Activation) (None, 14, 14, 816)  0           block5b_bn[0][0]                 
__________________________________________________________________________________________________
block5b_se_squeeze (GlobalAvera (None, 816)          0           block5b_activation[0][0]         
__________________________________________________________________________________________________
block5b_se_reshape (Reshape)    (None, 1, 1, 816)    0           block5b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5b_se_reduce (Conv2D)      (None, 1, 1, 34)     27778       block5b_se_reshape[0][0]         
__________________________________________________________________________________________________
block5b_se_expand (Conv2D)      (None, 1, 1, 816)    28560       block5b_se_reduce[0][0]          
__________________________________________________________________________________________________
block5b_se_excite (Multiply)    (None, 14, 14, 816)  0           block5b_activation[0][0]         
                                                                 block5b_se_expand[0][0]          
__________________________________________________________________________________________________
block5b_project_conv (Conv2D)   (None, 14, 14, 136)  110976      block5b_se_excite[0][0]          
__________________________________________________________________________________________________
block5b_project_bn (BatchNormal (None, 14, 14, 136)  544         block5b_project_conv[0][0]       
__________________________________________________________________________________________________
block5b_drop (Dropout)          (None, 14, 14, 136)  0           block5b_project_bn[0][0]         
__________________________________________________________________________________________________
block5b_add (Add)               (None, 14, 14, 136)  0           block5b_drop[0][0]               
                                                                 block5a_project_bn[0][0]         
__________________________________________________________________________________________________
block5c_expand_conv (Conv2D)    (None, 14, 14, 816)  110976      block5b_add[0][0]                
__________________________________________________________________________________________________
block5c_expand_bn (BatchNormali (None, 14, 14, 816)  3264        block5c_expand_conv[0][0]        
__________________________________________________________________________________________________
block5c_expand_activation (Acti (None, 14, 14, 816)  0           block5c_expand_bn[0][0]          
__________________________________________________________________________________________________
block5c_dwconv (DepthwiseConv2D (None, 14, 14, 816)  20400       block5c_expand_activation[0][0]  
__________________________________________________________________________________________________
block5c_bn (BatchNormalization) (None, 14, 14, 816)  3264        block5c_dwconv[0][0]             
__________________________________________________________________________________________________
block5c_activation (Activation) (None, 14, 14, 816)  0           block5c_bn[0][0]                 
__________________________________________________________________________________________________
block5c_se_squeeze (GlobalAvera (None, 816)          0           block5c_activation[0][0]         
__________________________________________________________________________________________________
block5c_se_reshape (Reshape)    (None, 1, 1, 816)    0           block5c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5c_se_reduce (Conv2D)      (None, 1, 1, 34)     27778       block5c_se_reshape[0][0]         
__________________________________________________________________________________________________
block5c_se_expand (Conv2D)      (None, 1, 1, 816)    28560       block5c_se_reduce[0][0]          
__________________________________________________________________________________________________
block5c_se_excite (Multiply)    (None, 14, 14, 816)  0           block5c_activation[0][0]         
                                                                 block5c_se_expand[0][0]          
__________________________________________________________________________________________________
block5c_project_conv (Conv2D)   (None, 14, 14, 136)  110976      block5c_se_excite[0][0]          
__________________________________________________________________________________________________
block5c_project_bn (BatchNormal (None, 14, 14, 136)  544         block5c_project_conv[0][0]       
__________________________________________________________________________________________________
block5c_drop (Dropout)          (None, 14, 14, 136)  0           block5c_project_bn[0][0]         
__________________________________________________________________________________________________
block5c_add (Add)               (None, 14, 14, 136)  0           block5c_drop[0][0]               
                                                                 block5b_add[0][0]                
__________________________________________________________________________________________________
block5d_expand_conv (Conv2D)    (None, 14, 14, 816)  110976      block5c_add[0][0]                
__________________________________________________________________________________________________
block5d_expand_bn (BatchNormali (None, 14, 14, 816)  3264        block5d_expand_conv[0][0]        
__________________________________________________________________________________________________
block5d_expand_activation (Acti (None, 14, 14, 816)  0           block5d_expand_bn[0][0]          
__________________________________________________________________________________________________
block5d_dwconv (DepthwiseConv2D (None, 14, 14, 816)  20400       block5d_expand_activation[0][0]  
__________________________________________________________________________________________________
block5d_bn (BatchNormalization) (None, 14, 14, 816)  3264        block5d_dwconv[0][0]             
__________________________________________________________________________________________________
block5d_activation (Activation) (None, 14, 14, 816)  0           block5d_bn[0][0]                 
__________________________________________________________________________________________________
block5d_se_squeeze (GlobalAvera (None, 816)          0           block5d_activation[0][0]         
__________________________________________________________________________________________________
block5d_se_reshape (Reshape)    (None, 1, 1, 816)    0           block5d_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5d_se_reduce (Conv2D)      (None, 1, 1, 34)     27778       block5d_se_reshape[0][0]         
__________________________________________________________________________________________________
block5d_se_expand (Conv2D)      (None, 1, 1, 816)    28560       block5d_se_reduce[0][0]          
__________________________________________________________________________________________________
block5d_se_excite (Multiply)    (None, 14, 14, 816)  0           block5d_activation[0][0]         
                                                                 block5d_se_expand[0][0]          
__________________________________________________________________________________________________
block5d_project_conv (Conv2D)   (None, 14, 14, 136)  110976      block5d_se_excite[0][0]          
__________________________________________________________________________________________________
block5d_project_bn (BatchNormal (None, 14, 14, 136)  544         block5d_project_conv[0][0]       
__________________________________________________________________________________________________
block5d_drop (Dropout)          (None, 14, 14, 136)  0           block5d_project_bn[0][0]         
__________________________________________________________________________________________________
block5d_add (Add)               (None, 14, 14, 136)  0           block5d_drop[0][0]               
                                                                 block5c_add[0][0]                
__________________________________________________________________________________________________
block5e_expand_conv (Conv2D)    (None, 14, 14, 816)  110976      block5d_add[0][0]                
__________________________________________________________________________________________________
block5e_expand_bn (BatchNormali (None, 14, 14, 816)  3264        block5e_expand_conv[0][0]        
__________________________________________________________________________________________________
block5e_expand_activation (Acti (None, 14, 14, 816)  0           block5e_expand_bn[0][0]          
__________________________________________________________________________________________________
block5e_dwconv (DepthwiseConv2D (None, 14, 14, 816)  20400       block5e_expand_activation[0][0]  
__________________________________________________________________________________________________
block5e_bn (BatchNormalization) (None, 14, 14, 816)  3264        block5e_dwconv[0][0]             
__________________________________________________________________________________________________
block5e_activation (Activation) (None, 14, 14, 816)  0           block5e_bn[0][0]                 
__________________________________________________________________________________________________
block5e_se_squeeze (GlobalAvera (None, 816)          0           block5e_activation[0][0]         
__________________________________________________________________________________________________
block5e_se_reshape (Reshape)    (None, 1, 1, 816)    0           block5e_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5e_se_reduce (Conv2D)      (None, 1, 1, 34)     27778       block5e_se_reshape[0][0]         
__________________________________________________________________________________________________
block5e_se_expand (Conv2D)      (None, 1, 1, 816)    28560       block5e_se_reduce[0][0]          
__________________________________________________________________________________________________
block5e_se_excite (Multiply)    (None, 14, 14, 816)  0           block5e_activation[0][0]         
                                                                 block5e_se_expand[0][0]          
__________________________________________________________________________________________________
block5e_project_conv (Conv2D)   (None, 14, 14, 136)  110976      block5e_se_excite[0][0]          
__________________________________________________________________________________________________
block5e_project_bn (BatchNormal (None, 14, 14, 136)  544         block5e_project_conv[0][0]       
__________________________________________________________________________________________________
block5e_drop (Dropout)          (None, 14, 14, 136)  0           block5e_project_bn[0][0]         
__________________________________________________________________________________________________
block5e_add (Add)               (None, 14, 14, 136)  0           block5e_drop[0][0]               
                                                                 block5d_add[0][0]                
__________________________________________________________________________________________________
block6a_expand_conv (Conv2D)    (None, 14, 14, 816)  110976      block5e_add[0][0]                
__________________________________________________________________________________________________
block6a_expand_bn (BatchNormali (None, 14, 14, 816)  3264        block6a_expand_conv[0][0]        
__________________________________________________________________________________________________
block6a_expand_activation (Acti (None, 14, 14, 816)  0           block6a_expand_bn[0][0]          
__________________________________________________________________________________________________
block6a_dwconv_pad (ZeroPadding (None, 17, 17, 816)  0           block6a_expand_activation[0][0]  
__________________________________________________________________________________________________
block6a_dwconv (DepthwiseConv2D (None, 7, 7, 816)    20400       block6a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block6a_bn (BatchNormalization) (None, 7, 7, 816)    3264        block6a_dwconv[0][0]             
__________________________________________________________________________________________________
block6a_activation (Activation) (None, 7, 7, 816)    0           block6a_bn[0][0]                 
__________________________________________________________________________________________________
block6a_se_squeeze (GlobalAvera (None, 816)          0           block6a_activation[0][0]         
__________________________________________________________________________________________________
block6a_se_reshape (Reshape)    (None, 1, 1, 816)    0           block6a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6a_se_reduce (Conv2D)      (None, 1, 1, 34)     27778       block6a_se_reshape[0][0]         
__________________________________________________________________________________________________
block6a_se_expand (Conv2D)      (None, 1, 1, 816)    28560       block6a_se_reduce[0][0]          
__________________________________________________________________________________________________
block6a_se_excite (Multiply)    (None, 7, 7, 816)    0           block6a_activation[0][0]         
                                                                 block6a_se_expand[0][0]          
__________________________________________________________________________________________________
block6a_project_conv (Conv2D)   (None, 7, 7, 232)    189312      block6a_se_excite[0][0]          
__________________________________________________________________________________________________
block6a_project_bn (BatchNormal (None, 7, 7, 232)    928         block6a_project_conv[0][0]       
__________________________________________________________________________________________________
block6b_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6a_project_bn[0][0]         
__________________________________________________________________________________________________
block6b_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block6b_expand_conv[0][0]        
__________________________________________________________________________________________________
block6b_expand_activation (Acti (None, 7, 7, 1392)   0           block6b_expand_bn[0][0]          
__________________________________________________________________________________________________
block6b_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   34800       block6b_expand_activation[0][0]  
__________________________________________________________________________________________________
block6b_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block6b_dwconv[0][0]             
__________________________________________________________________________________________________
block6b_activation (Activation) (None, 7, 7, 1392)   0           block6b_bn[0][0]                 
__________________________________________________________________________________________________
block6b_se_squeeze (GlobalAvera (None, 1392)         0           block6b_activation[0][0]         
__________________________________________________________________________________________________
block6b_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block6b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6b_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block6b_se_reshape[0][0]         
__________________________________________________________________________________________________
block6b_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block6b_se_reduce[0][0]          
__________________________________________________________________________________________________
block6b_se_excite (Multiply)    (None, 7, 7, 1392)   0           block6b_activation[0][0]         
                                                                 block6b_se_expand[0][0]          
__________________________________________________________________________________________________
block6b_project_conv (Conv2D)   (None, 7, 7, 232)    322944      block6b_se_excite[0][0]          
__________________________________________________________________________________________________
block6b_project_bn (BatchNormal (None, 7, 7, 232)    928         block6b_project_conv[0][0]       
__________________________________________________________________________________________________
block6b_drop (Dropout)          (None, 7, 7, 232)    0           block6b_project_bn[0][0]         
__________________________________________________________________________________________________
block6b_add (Add)               (None, 7, 7, 232)    0           block6b_drop[0][0]               
                                                                 block6a_project_bn[0][0]         
__________________________________________________________________________________________________
block6c_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6b_add[0][0]                
__________________________________________________________________________________________________
block6c_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block6c_expand_conv[0][0]        
__________________________________________________________________________________________________
block6c_expand_activation (Acti (None, 7, 7, 1392)   0           block6c_expand_bn[0][0]          
__________________________________________________________________________________________________
block6c_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   34800       block6c_expand_activation[0][0]  
__________________________________________________________________________________________________
block6c_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block6c_dwconv[0][0]             
__________________________________________________________________________________________________
block6c_activation (Activation) (None, 7, 7, 1392)   0           block6c_bn[0][0]                 
__________________________________________________________________________________________________
block6c_se_squeeze (GlobalAvera (None, 1392)         0           block6c_activation[0][0]         
__________________________________________________________________________________________________
block6c_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block6c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6c_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block6c_se_reshape[0][0]         
__________________________________________________________________________________________________
block6c_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block6c_se_reduce[0][0]          
__________________________________________________________________________________________________
block6c_se_excite (Multiply)    (None, 7, 7, 1392)   0           block6c_activation[0][0]         
                                                                 block6c_se_expand[0][0]          
__________________________________________________________________________________________________
block6c_project_conv (Conv2D)   (None, 7, 7, 232)    322944      block6c_se_excite[0][0]          
__________________________________________________________________________________________________
block6c_project_bn (BatchNormal (None, 7, 7, 232)    928         block6c_project_conv[0][0]       
__________________________________________________________________________________________________
block6c_drop (Dropout)          (None, 7, 7, 232)    0           block6c_project_bn[0][0]         
__________________________________________________________________________________________________
block6c_add (Add)               (None, 7, 7, 232)    0           block6c_drop[0][0]               
                                                                 block6b_add[0][0]                
__________________________________________________________________________________________________
block6d_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6c_add[0][0]                
__________________________________________________________________________________________________
block6d_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block6d_expand_conv[0][0]        
__________________________________________________________________________________________________
block6d_expand_activation (Acti (None, 7, 7, 1392)   0           block6d_expand_bn[0][0]          
__________________________________________________________________________________________________
block6d_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   34800       block6d_expand_activation[0][0]  
__________________________________________________________________________________________________
block6d_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block6d_dwconv[0][0]             
__________________________________________________________________________________________________
block6d_activation (Activation) (None, 7, 7, 1392)   0           block6d_bn[0][0]                 
__________________________________________________________________________________________________
block6d_se_squeeze (GlobalAvera (None, 1392)         0           block6d_activation[0][0]         
__________________________________________________________________________________________________
block6d_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block6d_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6d_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block6d_se_reshape[0][0]         
__________________________________________________________________________________________________
block6d_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block6d_se_reduce[0][0]          
__________________________________________________________________________________________________
block6d_se_excite (Multiply)    (None, 7, 7, 1392)   0           block6d_activation[0][0]         
                                                                 block6d_se_expand[0][0]          
__________________________________________________________________________________________________
block6d_project_conv (Conv2D)   (None, 7, 7, 232)    322944      block6d_se_excite[0][0]          
__________________________________________________________________________________________________
block6d_project_bn (BatchNormal (None, 7, 7, 232)    928         block6d_project_conv[0][0]       
__________________________________________________________________________________________________
block6d_drop (Dropout)          (None, 7, 7, 232)    0           block6d_project_bn[0][0]         
__________________________________________________________________________________________________
block6d_add (Add)               (None, 7, 7, 232)    0           block6d_drop[0][0]               
                                                                 block6c_add[0][0]                
__________________________________________________________________________________________________
block6e_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6d_add[0][0]                
__________________________________________________________________________________________________
block6e_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block6e_expand_conv[0][0]        
__________________________________________________________________________________________________
block6e_expand_activation (Acti (None, 7, 7, 1392)   0           block6e_expand_bn[0][0]          
__________________________________________________________________________________________________
block6e_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   34800       block6e_expand_activation[0][0]  
__________________________________________________________________________________________________
block6e_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block6e_dwconv[0][0]             
__________________________________________________________________________________________________
block6e_activation (Activation) (None, 7, 7, 1392)   0           block6e_bn[0][0]                 
__________________________________________________________________________________________________
block6e_se_squeeze (GlobalAvera (None, 1392)         0           block6e_activation[0][0]         
__________________________________________________________________________________________________
block6e_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block6e_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6e_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block6e_se_reshape[0][0]         
__________________________________________________________________________________________________
block6e_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block6e_se_reduce[0][0]          
__________________________________________________________________________________________________
block6e_se_excite (Multiply)    (None, 7, 7, 1392)   0           block6e_activation[0][0]         
                                                                 block6e_se_expand[0][0]          
__________________________________________________________________________________________________
block6e_project_conv (Conv2D)   (None, 7, 7, 232)    322944      block6e_se_excite[0][0]          
__________________________________________________________________________________________________
block6e_project_bn (BatchNormal (None, 7, 7, 232)    928         block6e_project_conv[0][0]       
__________________________________________________________________________________________________
block6e_drop (Dropout)          (None, 7, 7, 232)    0           block6e_project_bn[0][0]         
__________________________________________________________________________________________________
block6e_add (Add)               (None, 7, 7, 232)    0           block6e_drop[0][0]               
                                                                 block6d_add[0][0]                
__________________________________________________________________________________________________
block6f_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6e_add[0][0]                
__________________________________________________________________________________________________
block6f_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block6f_expand_conv[0][0]        
__________________________________________________________________________________________________
block6f_expand_activation (Acti (None, 7, 7, 1392)   0           block6f_expand_bn[0][0]          
__________________________________________________________________________________________________
block6f_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   34800       block6f_expand_activation[0][0]  
__________________________________________________________________________________________________
block6f_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block6f_dwconv[0][0]             
__________________________________________________________________________________________________
block6f_activation (Activation) (None, 7, 7, 1392)   0           block6f_bn[0][0]                 
__________________________________________________________________________________________________
block6f_se_squeeze (GlobalAvera (None, 1392)         0           block6f_activation[0][0]         
__________________________________________________________________________________________________
block6f_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block6f_se_squeeze[0][0]         
__________________________________________________________________________________________________
block6f_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block6f_se_reshape[0][0]         
__________________________________________________________________________________________________
block6f_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block6f_se_reduce[0][0]          
__________________________________________________________________________________________________
block6f_se_excite (Multiply)    (None, 7, 7, 1392)   0           block6f_activation[0][0]         
                                                                 block6f_se_expand[0][0]          
__________________________________________________________________________________________________
block6f_project_conv (Conv2D)   (None, 7, 7, 232)    322944      block6f_se_excite[0][0]          
__________________________________________________________________________________________________
block6f_project_bn (BatchNormal (None, 7, 7, 232)    928         block6f_project_conv[0][0]       
__________________________________________________________________________________________________
block6f_drop (Dropout)          (None, 7, 7, 232)    0           block6f_project_bn[0][0]         
__________________________________________________________________________________________________
block6f_add (Add)               (None, 7, 7, 232)    0           block6f_drop[0][0]               
                                                                 block6e_add[0][0]                
__________________________________________________________________________________________________
block7a_expand_conv (Conv2D)    (None, 7, 7, 1392)   322944      block6f_add[0][0]                
__________________________________________________________________________________________________
block7a_expand_bn (BatchNormali (None, 7, 7, 1392)   5568        block7a_expand_conv[0][0]        
__________________________________________________________________________________________________
block7a_expand_activation (Acti (None, 7, 7, 1392)   0           block7a_expand_bn[0][0]          
__________________________________________________________________________________________________
block7a_dwconv (DepthwiseConv2D (None, 7, 7, 1392)   12528       block7a_expand_activation[0][0]  
__________________________________________________________________________________________________
block7a_bn (BatchNormalization) (None, 7, 7, 1392)   5568        block7a_dwconv[0][0]             
__________________________________________________________________________________________________
block7a_activation (Activation) (None, 7, 7, 1392)   0           block7a_bn[0][0]                 
__________________________________________________________________________________________________
block7a_se_squeeze (GlobalAvera (None, 1392)         0           block7a_activation[0][0]         
__________________________________________________________________________________________________
block7a_se_reshape (Reshape)    (None, 1, 1, 1392)   0           block7a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block7a_se_reduce (Conv2D)      (None, 1, 1, 58)     80794       block7a_se_reshape[0][0]         
__________________________________________________________________________________________________
block7a_se_expand (Conv2D)      (None, 1, 1, 1392)   82128       block7a_se_reduce[0][0]          
__________________________________________________________________________________________________
block7a_se_excite (Multiply)    (None, 7, 7, 1392)   0           block7a_activation[0][0]         
                                                                 block7a_se_expand[0][0]          
__________________________________________________________________________________________________
block7a_project_conv (Conv2D)   (None, 7, 7, 384)    534528      block7a_se_excite[0][0]          
__________________________________________________________________________________________________
block7a_project_bn (BatchNormal (None, 7, 7, 384)    1536        block7a_project_conv[0][0]       
__________________________________________________________________________________________________
block7b_expand_conv (Conv2D)    (None, 7, 7, 2304)   884736      block7a_project_bn[0][0]         
__________________________________________________________________________________________________
block7b_expand_bn (BatchNormali (None, 7, 7, 2304)   9216        block7b_expand_conv[0][0]        
__________________________________________________________________________________________________
block7b_expand_activation (Acti (None, 7, 7, 2304)   0           block7b_expand_bn[0][0]          
__________________________________________________________________________________________________
block7b_dwconv (DepthwiseConv2D (None, 7, 7, 2304)   20736       block7b_expand_activation[0][0]  
__________________________________________________________________________________________________
block7b_bn (BatchNormalization) (None, 7, 7, 2304)   9216        block7b_dwconv[0][0]             
__________________________________________________________________________________________________
block7b_activation (Activation) (None, 7, 7, 2304)   0           block7b_bn[0][0]                 
__________________________________________________________________________________________________
block7b_se_squeeze (GlobalAvera (None, 2304)         0           block7b_activation[0][0]         
__________________________________________________________________________________________________
block7b_se_reshape (Reshape)    (None, 1, 1, 2304)   0           block7b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block7b_se_reduce (Conv2D)      (None, 1, 1, 96)     221280      block7b_se_reshape[0][0]         
__________________________________________________________________________________________________
block7b_se_expand (Conv2D)      (None, 1, 1, 2304)   223488      block7b_se_reduce[0][0]          
__________________________________________________________________________________________________
block7b_se_excite (Multiply)    (None, 7, 7, 2304)   0           block7b_activation[0][0]         
                                                                 block7b_se_expand[0][0]          
__________________________________________________________________________________________________
block7b_project_conv (Conv2D)   (None, 7, 7, 384)    884736      block7b_se_excite[0][0]          
__________________________________________________________________________________________________
block7b_project_bn (BatchNormal (None, 7, 7, 384)    1536        block7b_project_conv[0][0]       
__________________________________________________________________________________________________
block7b_drop (Dropout)          (None, 7, 7, 384)    0           block7b_project_bn[0][0]         
__________________________________________________________________________________________________
block7b_add (Add)               (None, 7, 7, 384)    0           block7b_drop[0][0]               
                                                                 block7a_project_bn[0][0]         
__________________________________________________________________________________________________
top_conv (Conv2D)               (None, 7, 7, 1536)   589824      block7b_add[0][0]                
__________________________________________________________________________________________________
top_bn (BatchNormalization)     (None, 7, 7, 1536)   6144        top_conv[0][0]                   
__________________________________________________________________________________________________
top_activation (Activation)     (None, 7, 7, 1536)   0           top_bn[0][0]                     
__________________________________________________________________________________________________
avg_pool (GlobalAveragePooling2 (None, 1536)         0           top_activation[0][0]             
__________________________________________________________________________________________________
pred_age (Dense)                (None, 101)          155237      avg_pool[0][0]                   
==================================================================================================
Total params: 10,938,772
Trainable params: 10,851,469
Non-trainable params: 87,303
__________________________________________________________________________________________________
>>>>>>>>>>>>>>>>>> START model.summary() >>>>>>>>>>>>>>>>>>

