// MIT License
//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY

// Preload for the splat-transform CLI (node --import <this file> cli.mjs ...).
//
// splat-transform ends every run with process.exit() while its cached WebGPU
// device is still alive ("something in webgpu seems to keep the process alive
// after returning from main so force exit" in its cli). Dawn (webgpu/dawn.node)
// then tears down a live Vulkan device from the exit handlers while its own
// threads are still running, which intermittently segfaults (exit code 139)
// AFTER the output was written and "done in ..." was printed.
//
// This preload records every GPUDevice the CLI creates and destroys them
// (device.destroy() waits for the GPU queue to go idle) before the real exit
// runs, so Dawn shuts down an idle, destroyed device instead.

import { createRequire, syncBuiltinESMExports } from 'node:module';
import { pathToFileURL } from 'node:url';

const devices = new Set();

try {
    // Resolve webgpu the same way the CLI does, so both share one dawn.node.
    const req = createRequire(process.argv[1] || import.meta.url);
    const { globals } = await import(pathToFileURL(req.resolve('webgpu')).href);
    const proto = globals?.GPUAdapter?.prototype;
    if (proto && typeof proto.requestDevice === 'function') {
        const requestDevice = proto.requestDevice;
        proto.requestDevice = async function (...args) {
            const device = await requestDevice.apply(this, args);
            if (device) {
                devices.add(device);
            }
            return device;
        };
    } else {
        process.stderr.write('[splat-transform-safe-exit] GPUAdapter.requestDevice not found; exit hook inactive\n');
    }
} catch (err) {
    process.stderr.write(`[splat-transform-safe-exit] could not hook webgpu: ${err?.message || err}\n`);
}

const destroyDevices = () => {
    if (devices.size === 0) {
        return;
    }
    for (const device of devices) {
        try {
            device.destroy();
        } catch {
            // One failing device must not stop the others from being destroyed.
        }
    }
    if (process.env.SPLAT_TRANSFORM_SAFE_EXIT_DEBUG) {
        process.stderr.write(`[splat-transform-safe-exit] destroyed ${devices.size} GPU device(s) before exit\n`);
    }
    devices.clear();
    // Give Dawn's and the driver's worker threads a moment to finish releasing
    // the destroyed devices before exit handlers run. Native threads keep
    // running while this thread sleeps.
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 250);
};

const realExit = process.exit;
process.exit = function exit(code) {
    destroyDevices();
    return realExit.call(process, code);
};
// The CLI uses `import { exit } from 'node:process'`; update that binding too.
syncBuiltinESMExports();

// Runs that end without process.exit() (normal event-loop drain).
process.once('beforeExit', destroyDevices);
