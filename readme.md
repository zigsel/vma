# vma

Idiomatic Zig bindings for the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).

Covers VMA's entire public surface — the allocator, resource creation, explicit
allocation, memory pages, custom pools, defragmentation, the virtual allocator,
statistics/budgets, corruption checks, aliasing and JSON dumps — in idiomatic
Zig: typed handles with methods, packed-struct flags, real enums, error unions,
slices and optionals. The raw C API is private; nothing public names a C type.

## Design notes

- `VMA_IMPLEMENTATION` is compiled **exactly once**, inside this package's static
  library (the translation unit is generated inline by `build.zig`). Consumers
  link it transitively — do **not** define `VMA_IMPLEMENTATION` yourself anywhere,
  or you'll get duplicate symbols.
- The dynamic-functions path is used: VMA loads everything through the
  `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr` pointers you pass to
  `Allocator.create`, so you don't have to link Vulkan statically.
- Flag bit positions and every re-interpreted struct layout are verified against
  the compiled VMA headers in a `comptime` block. If the headers drift, the build
  fails instead of corrupting memory.

### Will the dependencies collide with my engine?

No. Each Zig package has an isolated dependency tree, so this library fetching its
own Vulkan-Headers/VMA copies cannot clash with your engine's. C headers compiled
with `VK_NO_PROTOTYPES` emit no symbols, so duplicate header copies are harmless.
The only symbol-emitting translation unit is `VMA_IMPLEMENTATION`, which lives
here and only here. And because you supply the `vulkan` module, there is exactly
one vulkan-zig in your graph.

## Adding it to your project

```sh
zig fetch --save git+https://github.com/zigsel/vma
```

In your `build.zig`, grab the module and wire in *your* vulkan-zig module:

```zig
const vma = b.dependency("vma", .{
    .target = target,
    .optimize = optimize,
}).module("vma");

// Your existing vulkan-zig module:
const vulkan = b.dependency("vulkan_zig", .{
    .registry = some_vk_xml,
}).module("vulkan-zig");

vma.addImport("vulkan", vulkan);          // the one import the package leaves to you
exe.root_module.addImport("vma", vma);    // the C include paths + impl lib travel with it
```

## Usage

```zig
const vk = @import("vulkan");
const vma = @import("vma");

const allocator = try vma.Allocator.create(.{
    .instance = instance,
    .physical_device = physical_device,
    .device = device,
    .api_version = vk.API_VERSION_1_3,
    .get_instance_proc_addr = vkb.dispatch.vkGetInstanceProcAddr,
    .get_device_proc_addr = vki.dispatch.vkGetDeviceProcAddr,
    .flags = .{ .buffer_device_address = true },
});
defer allocator.destroy();

// Create a host-visible, mapped staging buffer in one call.
const staging = try allocator.createBuffer(&.{
    .size = 64 * 1024,
    .usage = .{ .transfer_src_bit = true },
    .sharing_mode = .exclusive,
}, .{
    .usage = .auto,
    .flags = .{ .mapped = true, .host_access_sequential_write = true },
});
defer allocator.destroyBuffer(staging.buffer, staging.allocation);

// `info.mapped_data` is non-null because of the `.mapped` flag.
const dst: [*]u8 = @ptrCast(staging.info.mapped_data.?);
@memcpy(dst[0..src.len], src);
```

See `src/root.zig` for the full, documented surface.

## Examples

- `examples/virtual_block.zig` — the virtual allocator (no GPU needed). Runnable:
  ```sh
  zig build run-virtual
  ```
- `examples/buffers.zig` — the device-backed allocator: vertex buffer, mapped
  staging upload, depth image, budgets and a stats dump. Needs a live device at
  runtime, so it's compile-checked rather than run.

Build them all with:

```sh
zig build examples
```

## Testing

```sh
zig build test
```

This compiles the binding against a private vulkan-zig copy and runs the
compile-time layout verification.
