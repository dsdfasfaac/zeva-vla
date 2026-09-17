#!/usr/bin/env python3
"""Enumerate Vulkan physical devices without creating a logical/render device.

This is an environment diagnostic only. It neither imports RoboTwin/SAPIEN nor
allocates a simulator scene, so it cannot establish formal renderer health.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import socket


class VkInstanceCreateInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("flags", C.c_uint32),
        ("pApplicationInfo", C.c_void_p),
        ("enabledLayerCount", C.c_uint32),
        ("ppEnabledLayerNames", C.c_void_p),
        ("enabledExtensionCount", C.c_uint32),
        ("ppEnabledExtensionNames", C.c_void_p),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, required=True)
    args = parser.parse_args()
    if not args.loader.is_file():
        raise FileNotFoundError(args.loader)
    lib = C.CDLL(str(args.loader))
    lib.vkCreateInstance.argtypes = [C.POINTER(VkInstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
    lib.vkCreateInstance.restype = C.c_int32
    lib.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p]
    lib.vkEnumeratePhysicalDevices.restype = C.c_int32
    lib.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
    lib.vkDestroyInstance.restype = None
    info = VkInstanceCreateInfo(sType=1)
    instance = C.c_void_p()
    create_result = lib.vkCreateInstance(C.byref(info), None, C.byref(instance))
    count = C.c_uint32()
    enumerate_result = None
    if create_result == 0:
        try:
            enumerate_result = lib.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
        finally:
            lib.vkDestroyInstance(instance, None)
    print(json.dumps({
        "schema": "zeva-vulkan-readonly-enumeration-v1",
        "host": socket.gethostname().split(".")[0],
        "loader": str(args.loader.resolve()),
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "create_result": int(create_result),
        "enumerate_result": None if enumerate_result is None else int(enumerate_result),
        "physical_device_count": int(count.value),
        "formal_renderer_health": False,
    }, sort_keys=True))
    if create_result != 0 or enumerate_result != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
