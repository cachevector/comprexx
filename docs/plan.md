# Comprexx Project Plan

## Compression Algorithms by File Type

### Universal Algorithms (all file types)

1. Huffman Coding (entropy coding)
2. Arithmetic Coding (better compression than Huffman)
3. LZ77/LZ78 (basis for many popular algorithms)
4. LZW (Lempel-Ziv-Welch) (used in GIF, TIFF)
5. Brotli (modern improvement on LZ77 + Huffman)
6. Zstandard (zstd) (modern, fast, good ratio)

### Text-Specific

1. Burrows-Wheeler Transform (BWT) + MTF + Huffman (used in bzip2)
2. PPM (Prediction by Partial Matching) (excellent for text)
3. Dictionary-based methods (especially for code)

### Image-Specific

1. RLE (Run-Length Encoding) (simple, good for simple images)
2. Delta Encoding (for certain image types)
3. DCT-based (JPEG style - lossy)
4. Wavelet transforms (JPEG2000, better than DCT)
5. Fractal Compression (research potential, though challenging)
6. PNG's DEFLATE (LZ77 + Huffman)

### Audio-Specific

1. Linear Predictive Coding (LPC)
2. Modified Discrete Cosine Transform (MDCT) (used in MP3)
3. FLAC's method (lossless, good for research)

### Binary/Executable

1. Context Mixing (PAQ series algorithms)
2. Bytewise Statistical Methods

### Experimental/Advanced

1. Neural Network-based Compression (research potential)
2. Genetic Algorithm-optimized compression
3. Data Deduplication techniques
4. ANS (Asymmetric Numeral Systems) - modern alternative to Huffman/Arithmetic

## Implementation Plan

1. Core Infrastructure:

- Metadata system to track compression methods used
- Benchmarking framework to compare algorithms
- File type detection system

2. Algorithm Implementation Roadmap:

- Phase 1: Implement basic algorithms (RLE, Huffman, LZ77)
- Phase 2: Specialized algorithms by file type
- Phase 3: Experimental methods (neural networks, genetic algorithms)
