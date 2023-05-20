## The Apple Neural Engine
Very little public information about it, and no public api afaik.
Seriously, searching for "site:machinelearning.apple.com apple neural engine" only gives information about 

[Gehot has reversed part of it](https://github.com/geohot/tinygrad/tree/master/accel/ane)

### [Apple Neural Engine Internal: From ML Algorithm to HW Registers](https://www.blackhat.com/asia-21/briefings/schedule/#apple-neural-engine-internal-from-ml-algorithm-to-hw-registers-22039)
Cool, so FaceId actually uses "Secure Neural Engine" and is also documented at [apples page on secure enclave](https://support.apple.com/lv-lv/guide/security/sec59b0b31ff/web).

They mention a tool, I have not heard of "Espresso" (*what is this ?* ).
So I found this page [A peek inside Core ML](https://machinethink.net/blog/peek-inside-coreml/) which concludes that Espresso is just a nickname for the part in [CoreML](https://developer.apple.com/documentation/coreml) that runs neural networks.

The general pipeline is described as `CoreML -> Espresso -> Apple Neural Engine Compiler (APE)`, but I have not been able to find any information about the APE. It's however discussed some in the talk (currently only looked at the slides).

Cool, they have released the tools they used [https://github.com/antgroup-arclab/ANETools](https://github.com/antgroup-arclab/ANETools) 
