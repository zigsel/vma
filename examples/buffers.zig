//! Compile-checked example: the device-backed allocator and resource creation.
//!
//! This path needs a live `VkInstance`/`VkPhysicalDevice`/`VkDevice` and the two
//! Vulkan proc-address loaders, so it can't run in CI — but it is built by
//! `zig build examples`, so the usage below is always kept compiling against the
//! real API. Drop the bodies into your renderer and pass your own handles.

const std = @import("std");
const vk = @import("vulkan");
const vma = @import("vma");

/// Everything you need to stand up the allocator.
pub const Context = struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    get_instance_proc_addr: vk.PfnGetInstanceProcAddr,
    get_device_proc_addr: vk.PfnGetDeviceProcAddr,
};

pub fn createAllocator(ctx: Context) !vma.Allocator {
    return vma.Allocator.create(.{
        .instance = ctx.instance,
        .physical_device = ctx.physical_device,
        .device = ctx.device,
        .api_version = vk.API_VERSION_1_3,
        .get_instance_proc_addr = ctx.get_instance_proc_addr,
        .get_device_proc_addr = ctx.get_device_proc_addr,
    });
}

/// A device-local vertex buffer, allocated and bound in one call.
pub fn createVertexBuffer(allocator: vma.Allocator, size: vk.DeviceSize) !vma.Allocator.BufferResult {
    return allocator.createBuffer(&.{
        .size = size,
        .usage = .{ .vertex_buffer_bit = true, .transfer_dst_bit = true },
        .sharing_mode = .exclusive,
    }, .{
        .usage = .auto,
    });
}

/// A host-visible, persistently mapped staging buffer, then copy `data` into it.
pub fn uploadViaStaging(allocator: vma.Allocator, data: []const u8) !vma.Allocator.BufferResult {
    const staging = try allocator.createBuffer(&.{
        .size = data.len,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, .{
        .usage = .auto,
        .flags = .{ .mapped = true, .host_access_sequential_write = true },
    });

    // `.mapped` guarantees a stable pointer for the lifetime of the allocation.
    const dst: [*]u8 = @ptrCast(staging.info.mapped_data.?);
    @memcpy(dst[0..data.len], data);

    // Non-coherent memory needs an explicit flush; harmless on coherent memory.
    try allocator.flushAllocation(staging.allocation, 0, data.len);
    return staging;
}

/// A depth image placed in device-local memory.
pub fn createDepthImage(allocator: vma.Allocator, width: u32, height: u32) !vma.Allocator.ImageResult {
    return allocator.createImage(&.{
        .image_type = .@"2d",
        .format = .d32_sfloat,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{ .depth_stencil_attachment_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, .{
        .usage = .auto,
        .priority = 1.0,
    });
}

/// End-to-end shape, showing the matching destroy calls.
pub fn demo(ctx: Context) !void {
    const allocator = try createAllocator(ctx);
    defer allocator.destroy();

    const vbuf = try createVertexBuffer(allocator, 4096);
    defer allocator.destroyBuffer(vbuf.buffer, vbuf.allocation);

    const staging = try uploadViaStaging(allocator, "hello vma");
    defer allocator.destroyBuffer(staging.buffer, staging.allocation);

    const depth = try createDepthImage(allocator, 1920, 1080);
    defer allocator.destroyImage(depth.image, depth.allocation);

    // Per-heap budgets, sized exactly to the device's heap count.
    var budgets: [vk.MAX_MEMORY_HEAPS]vma.Budget = undefined;
    for (allocator.getHeapBudgets(&budgets), 0..) |bud, heap| {
        std.debug.print("heap {d}: {d}/{d} bytes\n", .{ heap, bud.usage, bud.budget });
    }

    const json = allocator.buildStatsString(true);
    defer allocator.freeStatsString(json);
    std.debug.print("{s}\n", .{json});
}

// `main` exists so this file builds as an executable. It performs no Vulkan calls
// (there's no device here); it only forces the example functions to compile.
pub fn main() void {
    inline for (.{ createAllocator, createVertexBuffer, uploadViaStaging, createDepthImage, demo }) |f| {
        _ = &f;
    }
    std.debug.print("buffers.zig is a compile-only example; see its source.\n", .{});
}
