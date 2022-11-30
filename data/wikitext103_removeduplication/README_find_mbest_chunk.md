512 * 128

最佳chunk是128

1. chunk_size: 128 * 512: 0.3784%|
2. chunk_size: 256 * 256: 0.3595%
3. chunk_size: 512 * 128: 0.3174%

确定最佳chunk 128,找最佳token数
1. chunk_size: 128, 64: 24.9
1. chunk_size: 128, 128: 29.18
1. chunk_size: 128, 256: 33.53%
1. chunk_size: 128, 512: 37.84%
1. chunk_size: 128, 1024: 41.53%


