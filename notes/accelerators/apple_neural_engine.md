## The Apple Neural Engine
Very little public information about it, and no public api afaik.
Seriously, searching for "site:machinelearning.apple.com apple neural engine" only gives information about 

[Gehot has reversed part of it](https://github.com/tinygrad/tinygrad/tree/a8f2c16f8e1670ce199b068a771b9b0d6f7ba7df/extra/accel/ane)

### [Apple Neural Engine Internal: From ML Algorithm to HW Registers](https://www.blackhat.com/asia-21/briefings/schedule/#apple-neural-engine-internal-from-ml-algorithm-to-hw-registers-22039)
Cool, so FaceId actually uses "Secure Neural Engine" and is also documented at [apples page on secure enclave](https://support.apple.com/lv-lv/guide/security/sec59b0b31ff/web).

They mention a tool, I have not heard of "Espresso" (*what is this ?* ).
So I found this page [A peek inside Core ML](https://machinethink.net/blog/peek-inside-coreml/) which concludes that Espresso is just a nickname for the part in [CoreML](https://developer.apple.com/documentation/coreml) that runs neural networks.

The general pipeline is described as `CoreML -> Espresso -> Apple Neural Engine Compiler (APE)`, but I have not been able to find any information about the APE. It's however discussed some in the talk (currently only looked at the slides).

Cool, they have released the tools they used [https://github.com/antgroup-arclab/ANETools](https://github.com/antgroup-arclab/ANETools) 

### [Core ML](https://developer.apple.com/documentation/coreml)
Apple's framework for how to deploy ML interference on the device.

[Core ML Tools Overview](https://coremltools.readme.io/docs)

[https://machinethink.net/blog/](https://machinethink.net/blog/) has some good blogposts on the overview of Core ML


### [AMX](https://github.com/corsix/amx)
The instruction set used for the Apple Matrix Coprocessor. This is not the same as the neural engine.

It's not the same as SIMD Vector Engine, but closer to that then the Neural engine chip.


[The Secret Apple M1 Coprocessor](https://web.archive.org/web/20210206122953/https://medium.com/swlh/apples-m1-secret-coprocessor-6599492fc1e1) 

[  BLIS & TBLIS for the Undocumented Apple Matrix Coprocessor - BLIS Retreat2021 Long Version ](https://www.youtube.com/watch?v=HpgRxT3m80U)

[BLIS fork with kernels for Apple M1](https://github.com/xrq-phys/blis_apple)
