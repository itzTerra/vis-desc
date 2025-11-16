#!/usr/bin/env python3
"""
GPU Information Script
Prints comprehensive information about CUDA and GPU devices.
"""

import torch
import os
import sys


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_cuda_availability():
    """Print CUDA availability information."""
    print_section("CUDA Availability")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")


def print_device_info():
    """Print detailed information about each GPU device."""
    print_section("GPU Devices")
    try:
        device_count = torch.cuda.device_count()
    except Exception as e:
        print(f"Error getting device count: {e}\nNo CUDA devices available.")
    print(f"Number of GPUs: {device_count}\n")

    for i in range(device_count):
        print(f"--- Device {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
        print()


def print_current_device():
    """Print information about the currently selected device."""
    print_section("Current Device Selection")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current Device ID: {current_device}")
        print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No CUDA device selected (CPU mode)")

    # Show the device used by default torch.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Default torch.device: {device}")


def print_environment_variables():
    """Print relevant CUDA environment variables."""
    print_section("Environment Variables")

    cuda_vars = {
        "CUDA_VISIBLE_DEVICES": "Controls which GPUs are visible to CUDA",
        "CUDA_DEVICE_ORDER": "Device enumeration order (PCI_BUS_ID or FASTEST_FIRST)",
        "CUDA_LAUNCH_BLOCKING": "Synchronous CUDA operations for debugging",
        "CUDA_CACHE_DISABLE": "Disable CUDA kernel caching",
        "CUDA_HOME": "CUDA installation directory",
        "CUDA_PATH": "Alternative CUDA installation path",
    }

    for var, description in cuda_vars.items():
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
        print(f"  ({description})")
        print()


def print_device_order():
    """Print information about device ordering."""
    print_section("Device Ordering")
    device_order = os.environ.get(
        "CUDA_DEVICE_ORDER", "Not set (default: FASTEST_FIRST)"
    )
    print(f"CUDA_DEVICE_ORDER: {device_order}")
    print("\nDevice Order Options:")
    print("  - FASTEST_FIRST: Orders devices by compute capability (default)")
    print("  - PCI_BUS_ID: Orders devices by PCI bus ID (matches nvidia-smi)")

    if torch.cuda.is_available():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "All devices visible")
        print(f"\nCUDA_VISIBLE_DEVICES: {visible_devices}")


def print_misc_info():
    """Print miscellaneous GPU-related information."""
    print_section("Miscellaneous")

    if torch.cuda.is_available():
        print(f"CUDA Initialized: {torch.cuda.is_initialized()}")
        print(f"Default Stream: {torch.cuda.default_stream()}")
        print("Number of Streams: Can create unlimited streams")

        # Arch list
        print(f"\nSupported CUDA Architectures: {torch.cuda.get_arch_list()}")

        # Check if specific GPU features are available
        print(
            f"\nTensor Cores Available: {any(torch.cuda.get_device_capability(i)[0] >= 7 for i in range(torch.cuda.device_count()))}"
        )


def main():
    """Main function to print all GPU information."""
    print("\n" + "=" * 60)
    print("  GPU Information Report")
    print("=" * 60)

    try:
        print_cuda_availability()
        print_device_info()
        print_current_device()
        print_environment_variables()
        print_device_order()
        print_misc_info()

        print("\n" + "=" * 60)
        print("  End of Report")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
