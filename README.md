<div align="center">

# Native Multi-Band Audio Coding within Hyper-Autoencoded Reconstruction Propogation Networks

[Darius Pétermann](https://www.dariuspetermann.com)  and  [Minje Kim](https://saige.sice.indiana.edu)

[![Demo](https://img.shields.io/badge/Web-Demo-blue)](https://darius522.github.io/mb-harpnet/)

<br>

<img src="docs/images/overview.png" width='1000'>>

</div>

## Abstract
Spectral sub-bands do not portray the same perceptual relevance. In audio coding, it is therefore desirable to have independent control over each of the constitutive bands so that bitrate assignment and signal reconstruction can be achieved efficiently. In this work, we present a novel neural audio coding network that natively supports a multi-band coding paradigm. Our model extends the idea of com- pressed skip connections in the U-Net-based codec, allowing for in- dependent control over both core and high band-specific reconstruc- tions and bitrate assignments. Our system reconstructs the full-band signal mainly from the condensed core-band code, therefore exploit- ing and showcasing its bandwidth extension capabilities to its fullest. Meanwhile, the high-band code sends a small number of bits to help the high-band reconstruction similarly to MPEG audio codec’s spec- tral bandwidth replication. Through MUSHRA tests, we show that the proposed model not only improves the quality of the core band by explicitly assigning more bits to it but retains a good quality in the high-band as well.

<br>

<div align="center">