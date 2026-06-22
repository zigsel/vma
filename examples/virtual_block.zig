//! Runnable example: VMA's *virtual allocator*.
//!
//! A `VirtualBlock` runs VMA's allocation algorithm over a plain integer address
//! space — no Vulkan device, no GPU memory. It's handy for sub-allocating your
//! own buffers/arenas. Because it touches no Vulkan, this example really runs:
//!
//!     zig build run-virtual
//!
//! It carves a 1 MiB space into a few sub-allocations, prints their offsets,
//! frees one, shows the freed room is reused, and dumps the block statistics.

const std = @import("std");
const vma = @import("vma");

pub fn main() !void {
    const mib = 1024 * 1024;

    var block = try vma.VirtualBlock.create(.{ .size = mib });
    defer block.destroy();

    // Three sub-allocations with different sizes and alignments.
    const a = try block.allocate(.{ .size = 256 * 1024, .alignment = 256 });
    const b = try block.allocate(.{ .size = 128 * 1024, .alignment = 256 });
    const d = try block.allocate(.{ .size = 512 * 1024, .alignment = 4096 });

    std.debug.print("a.offset = {d}\n", .{a.offset});
    std.debug.print("b.offset = {d}\n", .{b.offset});
    std.debug.print("d.offset = {d}\n", .{d.offset});

    // Free the middle one; its range becomes available again.
    block.free(b.allocation);

    const e = try block.allocate(.{ .size = 64 * 1024, .alignment = 256 });
    std.debug.print("e.offset = {d} (reused b's region)\n", .{e.offset});

    // User data round-trips through the allocation.
    var tag: u32 = 0xC0FFEE;
    block.setAllocationUserData(e.allocation, &tag);
    const info = block.getAllocationInfo(e.allocation);
    std.debug.assert(info.user_data == @as(*anyopaque, &tag));
    std.debug.assert(info.size == 64 * 1024);

    const stats = block.getStatistics();
    std.debug.print(
        "\nblocks={d} allocations={d} used={d} of {d} bytes\n",
        .{ stats.block_count, stats.allocation_count, stats.allocation_bytes, stats.block_bytes },
    );

    // Clearing drops every remaining allocation at once.
    block.clear();
    std.debug.assert(block.isEmpty());
    std.debug.print("cleared; block empty = {}\n", .{block.isEmpty()});
}
