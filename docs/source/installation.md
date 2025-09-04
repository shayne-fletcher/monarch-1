## Installation

### Prerequisites

Before installing Monarch, ensure you have:

- A Linux system (Monarch is currently only supported on Linux)
- Python 3.10 or later
- CUDA-compatible GPU(s)
- Basic familiarity with PyTorch


### Quick Installation

The simplest way to install Monarch is via pip:

```bash
pip install torchmonarch-nightly
```

### Manual Installation

For more control or development purposes, you can install Monarch manually:

```bash
# Create and activate the conda environment
conda create -n monarchenv python=3.10 -y
conda activate monarchenv

# Install nightly rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly

# Install non-python dependencies
conda install libunwind -y

# Install the correct cuda and cuda-toolkit versions for your machine
sudo dnf install cuda-toolkit-12-0 cuda-12-0

# Install clang-dev and nccl-dev
sudo dnf install clang-devel libnccl-devel
# Or, in some environments, the following may be necessary instead
conda install -c conda-forge clangdev nccl
conda update -n monarchenv --all -c conda-forge -y

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch
pip install --no-build-isolation .
# or setup for development
pip install --no-build-isolation -e .
```

## Verifying Your Installation

After installation, you can verify that Monarch is working correctly by running the unit tests:

```bash
pytest python/tests/ -v -m "not oss_skip"
```

## Next Steps

Now that you've got the basics, you can:

1. Read the [getting started](./generated/examples/getting_started) guide to understand the core concepts.
1. Check out the [Examples](./generated/examples/index) directory for more detailed demonstrations
2. Explore the [API documentation](python-api) for a complete reference

## Troubleshooting

If you encounter issues:

- Make sure your CUDA environment is properly set up
- Check that you're using a compatible version of PyTorch
- Verify that all dependencies are installed correctly
- Consult the [GitHub repository](https://github.com/meta-pytorch/monarch) for known issues

Remember that Monarch is currently in an experimental stage, so you may encounter bugs or incomplete features. Contributions and bug reports are welcome!
