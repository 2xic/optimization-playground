## Tesla Dojo
So Dojo is the name of the super computer that tesla uses for training, but it's based on their own chip (D1) so that makes it worth exploring.

# [Hot Chips 34 – Tesla’s Dojo Microarchitecture](https://chipsandcheese.com/2022/09/01/hot-chips-34-teslas-dojo-microarchitecture/)
- RISC-V like architecture with some custom instructions for vectors.
- Tesla does various tradeoffs that makes life harder for people working with the chip, but the gain is that they get more performance.

# [Super-Compute System Scaling for ML Training (hot chips presentation)](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/Hotchip%20Dojo%20System%20v25.pdf)
- Uses custom transfer protocols (Tesla Transport Protocol) to have a good split between bandwidth and latency

# [The Microarchitecture of Tesla’s Exa-Scale Computer (hot chips presentation)](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)
- *TODO*

# [Tesla’s Dojo Supercomputer Deep Dive](https://web.archive.org/web/20221218050943/https://morethanmoore.substack.com/p/teslas-dojo-supercomputer-deep-dive)
- Built with TSMC's N7 process
- Covers the hot chips presentations with some additional context
- *todo* This is actually a good blogpost need to look more into it

# [Tesla Dojo Technology](https://cdn.motor1.com/pdf-files/535242876-tesla-dojo-technology.pdf)
- This only documents the floating points used. 

# [Tesla’s TTPoE at Hot Chips 2024: Replacing TCP for Low Latency Applications](https://chipsandcheese.com/2024/08/27/teslas-ttpoe-at-hot-chips-2024-replacing-tcp-for-low-latency-applications/)
- The size of a single tensor could be 1.7 GB, this is a bit ambiguous though.
- They noticed a problem that computers could be slow at transferring data even if all they do is transfer over PCI
  - Tesla solved it by adding more host
  - They didn't use [Infiniband](https://en.wikipedia.org/wiki/InfiniBand) like many others. Instead they do Ethernet and a custom transport layer. [Infiniband tax](https://news.ycombinator.com/item?id=41379866) escape possibly.
  - ^ Latency is reduced by removing wait status in TCP
- [I found the slides](https://hc2024.hotchips.org/assets/program/conference/day2/17_HC2024_Tesla_TTPoE_v5.pdf), but they are password protected. Found them on some random chinese websites also, but also no way to download them.
  - nvm the speaker [had posted about it on Twitter](https://x.com/divBy_zero/status/1830441307594174496)
- 
