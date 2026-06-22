//! Idiomatic Zig bindings for the Vulkan Memory Allocator (VMA).
//!
//! This covers VMA's entire public surface — the allocator, resource creation,
//! explicit allocation, memory pages, custom pools, defragmentation, the virtual
//! allocator, statistics/budgets, corruption checks, aliasing and JSON dumps —
//! expressed in idiomatic Zig: typed handles with methods, packed-struct flags,
//! real enums, error unions, slices and optionals.
//!
//! The raw C API is kept private; nothing in the public surface names a C type.
//! Vulkan objects are taken and returned as vulkan-zig types
//! whose memory layout is identical to the C structs, so they cross the boundary
//! by pointer reinterpretation.
//!
//! Two robustness invariants are enforced at compile time (see the bottom of the
//! file): every packed-struct flag bit is checked against the corresponding VMA
//! `*_BIT` constant, and every re-interpreted struct is checked to have the same
//! `@sizeOf` as its C counterpart. If VMA's headers ever shift, the build breaks
//! instead of silently corrupting memory.
//!
//! Platform note: VMA's Win32 external-memory helper `vmaGetMemoryWin32Handle`
//! only exists under `VK_USE_PLATFORM_WIN32_KHR`; it is absent on this build and
//! therefore not bound.

const std = @import("std");
const vk = @import("vulkan");
const assert = std.debug.assert;

// The private C view of VMA. Produced by `b.addTranslateC` in build.zig.
const c = @import("vma_h");

// ===========================================================================
// Errors
// ===========================================================================

pub const Error = error{
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    MemoryMapFailed,
    DeviceLost,
    FeatureNotPresent,
    TooManyObjects,
    FormatNotSupported,
    FragmentedPool,
    OutOfPoolMemory,
    InvalidExternalHandle,
    Unknown,
};

fn check(r: c.VkResult) Error!void {
    return switch (r) {
        c.VK_SUCCESS => {},
        c.VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        c.VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        c.VK_ERROR_MEMORY_MAP_FAILED => error.MemoryMapFailed,
        c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
        c.VK_ERROR_FEATURE_NOT_PRESENT => error.FeatureNotPresent,
        c.VK_ERROR_TOO_MANY_OBJECTS => error.TooManyObjects,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => error.FormatNotSupported,
        c.VK_ERROR_FRAGMENTED_POOL => error.FragmentedPool,
        c.VK_ERROR_OUT_OF_POOL_MEMORY => error.OutOfPoolMemory,
        c.VK_ERROR_INVALID_EXTERNAL_HANDLE => error.InvalidExternalHandle,
        else => error.Unknown,
    };
}

// ===========================================================================
// Handle conversion (vulkan-zig enum handle <-> C opaque pointer)
// ===========================================================================

/// vulkan-zig represents Vulkan handles as `enum(usize)`/`enum(u64)` values; the
/// C API wants opaque pointers. Both are just the same integer, so we convert by
/// round-tripping through it. A null handle (value 0) becomes a null pointer.
inline fn toC(comptime CHandle: type, handle: anytype) CHandle {
    return @ptrFromInt(@intFromEnum(handle));
}

inline fn toVk(comptime VkHandle: type, ptr: anytype) VkHandle {
    return @enumFromInt(@intFromPtr(ptr));
}

inline fn optName(p: [*c]const u8) ?[*:0]const u8 {
    return if (p == null) null else @ptrCast(p);
}

inline fn vkFlags(comptime T: type, raw: u32) T {
    return @bitCast(raw);
}

// ===========================================================================
// Enums
// ===========================================================================

pub const MemoryUsage = enum(c_int) {
    unknown = c.VMA_MEMORY_USAGE_UNKNOWN,
    gpu_only = c.VMA_MEMORY_USAGE_GPU_ONLY,
    cpu_only = c.VMA_MEMORY_USAGE_CPU_ONLY,
    cpu_to_gpu = c.VMA_MEMORY_USAGE_CPU_TO_GPU,
    gpu_to_cpu = c.VMA_MEMORY_USAGE_GPU_TO_CPU,
    cpu_copy = c.VMA_MEMORY_USAGE_CPU_COPY,
    gpu_lazily_allocated = c.VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED,
    auto = c.VMA_MEMORY_USAGE_AUTO,
    auto_prefer_device = c.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    auto_prefer_host = c.VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
};

pub const DefragmentationMoveOperation = enum(c_int) {
    /// Buffer/image has been recreated at `dstTmp`; copy and replace the handle.
    copy = c.VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY,
    /// Leave the allocation in place this pass.
    ignore = c.VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE,
    /// The allocation is no longer needed; destroy it.
    destroy = c.VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY,
};

// ===========================================================================
// Flags (packed structs over u32, matching VMA bit positions)
// ===========================================================================

pub const AllocatorCreateFlags = packed struct(u32) {
    externally_synchronized: bool = false, // 0x0001
    khr_dedicated_allocation: bool = false, // 0x0002
    khr_bind_memory2: bool = false, // 0x0004
    ext_memory_budget: bool = false, // 0x0008
    amd_device_coherent_memory: bool = false, // 0x0010
    buffer_device_address: bool = false, // 0x0020
    ext_memory_priority: bool = false, // 0x0040
    khr_maintenance4: bool = false, // 0x0080
    khr_maintenance5: bool = false, // 0x0100
    khr_external_memory_win32: bool = false, // 0x0200
    _reserved: u22 = 0,

    pub inline fn toInt(self: AllocatorCreateFlags) u32 {
        return @bitCast(self);
    }
};

pub const AllocationCreateFlags = packed struct(u32) {
    dedicated_memory: bool = false, // 0x0001
    never_allocate: bool = false, // 0x0002
    mapped: bool = false, // 0x0004
    _r3_4: u2 = 0,
    user_data_copy_string: bool = false, // 0x0020
    upper_address: bool = false, // 0x0040
    dont_bind: bool = false, // 0x0080
    within_budget: bool = false, // 0x0100
    can_alias: bool = false, // 0x0200
    host_access_sequential_write: bool = false, // 0x0400
    host_access_random: bool = false, // 0x0800
    host_access_allow_transfer_instead: bool = false, // 0x1000
    _r13_15: u3 = 0,
    /// Also the BEST_FIT strategy.
    strategy_min_memory: bool = false, // 0x10000
    /// Also the FIRST_FIT strategy.
    strategy_min_time: bool = false, // 0x20000
    strategy_min_offset: bool = false, // 0x40000
    _reserved: u13 = 0,

    pub inline fn toInt(self: AllocationCreateFlags) u32 {
        return @bitCast(self);
    }
};

pub const PoolCreateFlags = packed struct(u32) {
    _r0: u1 = 0,
    ignore_buffer_image_granularity: bool = false, // 0x0002
    linear_algorithm: bool = false, // 0x0004
    _reserved: u29 = 0,

    pub inline fn toInt(self: PoolCreateFlags) u32 {
        return @bitCast(self);
    }
};

pub const DefragmentationFlags = packed struct(u32) {
    algorithm_fast: bool = false, // 0x1
    algorithm_balanced: bool = false, // 0x2
    algorithm_full: bool = false, // 0x4
    algorithm_extensive: bool = false, // 0x8
    _reserved: u28 = 0,

    pub inline fn toInt(self: DefragmentationFlags) u32 {
        return @bitCast(self);
    }
};

pub const VirtualBlockCreateFlags = packed struct(u32) {
    linear_algorithm: bool = false, // 0x1
    _reserved: u31 = 0,

    pub inline fn toInt(self: VirtualBlockCreateFlags) u32 {
        return @bitCast(self);
    }
};

pub const VirtualAllocationCreateFlags = packed struct(u32) {
    _r0_5: u6 = 0,
    upper_address: bool = false, // 0x40
    _r7_15: u9 = 0,
    strategy_min_memory: bool = false, // 0x10000
    strategy_min_time: bool = false, // 0x20000
    strategy_min_offset: bool = false, // 0x40000
    _reserved: u13 = 0,

    pub inline fn toInt(self: VirtualAllocationCreateFlags) u32 {
        return @bitCast(self);
    }
};

// ===========================================================================
// Plain value structs (idiomatic mirrors of VMA's POD structs)
// ===========================================================================

pub const AllocatorInfo = struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
};

pub const AllocationInfo = struct {
    memory_type: u32,
    device_memory: vk.DeviceMemory,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    mapped_data: ?*anyopaque,
    user_data: ?*anyopaque,
    name: ?[*:0]const u8,

    fn from(ci: c.VmaAllocationInfo) AllocationInfo {
        return .{
            .memory_type = ci.memoryType,
            .device_memory = toVk(vk.DeviceMemory, ci.deviceMemory),
            .offset = ci.offset,
            .size = ci.size,
            .mapped_data = ci.pMappedData,
            .user_data = ci.pUserData,
            .name = optName(ci.pName),
        };
    }
};

pub const AllocationInfo2 = struct {
    info: AllocationInfo,
    block_size: vk.DeviceSize,
    dedicated_memory: bool,
};

pub const Statistics = struct {
    block_count: u32,
    allocation_count: u32,
    block_bytes: vk.DeviceSize,
    allocation_bytes: vk.DeviceSize,

    fn from(s: c.VmaStatistics) Statistics {
        return .{
            .block_count = s.blockCount,
            .allocation_count = s.allocationCount,
            .block_bytes = s.blockBytes,
            .allocation_bytes = s.allocationBytes,
        };
    }
};

pub const DetailedStatistics = struct {
    statistics: Statistics,
    unused_range_count: u32,
    allocation_size_min: vk.DeviceSize,
    allocation_size_max: vk.DeviceSize,
    unused_range_size_min: vk.DeviceSize,
    unused_range_size_max: vk.DeviceSize,

    fn from(d: c.VmaDetailedStatistics) DetailedStatistics {
        return .{
            .statistics = Statistics.from(d.statistics),
            .unused_range_count = d.unusedRangeCount,
            .allocation_size_min = d.allocationSizeMin,
            .allocation_size_max = d.allocationSizeMax,
            .unused_range_size_min = d.unusedRangeSizeMin,
            .unused_range_size_max = d.unusedRangeSizeMax,
        };
    }
};

pub const TotalStatistics = struct {
    memory_type: [vk.MAX_MEMORY_TYPES]DetailedStatistics,
    memory_heap: [vk.MAX_MEMORY_HEAPS]DetailedStatistics,
    total: DetailedStatistics,

    fn from(t: c.VmaTotalStatistics) TotalStatistics {
        var out: TotalStatistics = undefined;
        for (&out.memory_type, t.memoryType[0..vk.MAX_MEMORY_TYPES]) |*o, s| o.* = DetailedStatistics.from(s);
        for (&out.memory_heap, t.memoryHeap[0..vk.MAX_MEMORY_HEAPS]) |*o, s| o.* = DetailedStatistics.from(s);
        out.total = DetailedStatistics.from(t.total);
        return out;
    }
};

pub const Budget = struct {
    statistics: Statistics,
    usage: vk.DeviceSize,
    budget: vk.DeviceSize,

    fn from(b: c.VmaBudget) Budget {
        return .{
            .statistics = Statistics.from(b.statistics),
            .usage = b.usage,
            .budget = b.budget,
        };
    }
};

pub const DefragmentationStats = struct {
    bytes_moved: vk.DeviceSize,
    bytes_freed: vk.DeviceSize,
    allocations_moved: u32,
    device_memory_blocks_freed: u32,
};

pub const VirtualAllocationInfo = struct {
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    user_data: ?*anyopaque,
};

// --- Device-memory callbacks ------------------------------------------------
//
// The allocator handle is delivered to callbacks as a raw pointer to avoid
// struct-by-value ABI assumptions; wrap it with `Allocator{ .handle = ... }` if
// you need methods.

pub const AllocateDeviceMemoryFn = *const fn (
    allocator: ?*anyopaque,
    memory_type: u32,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    user_data: ?*anyopaque,
) callconv(.c) void;

pub const FreeDeviceMemoryFn = *const fn (
    allocator: ?*anyopaque,
    memory_type: u32,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    user_data: ?*anyopaque,
) callconv(.c) void;

pub const DeviceMemoryCallbacks = struct {
    allocate: ?AllocateDeviceMemoryFn = null,
    free: ?FreeDeviceMemoryFn = null,
    user_data: ?*anyopaque = null,
};

/// A break callback returns `vk.TRUE` to stop defragmentation early.
pub const CheckDefragmentationBreakFn = *const fn (user_data: ?*anyopaque) callconv(.c) vk.Bool32;

// ===========================================================================
// Create-info structs (idiomatic inputs)
// ===========================================================================

pub const CreateInfo = struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    api_version: vk.Version,
    get_instance_proc_addr: vk.PfnGetInstanceProcAddr,
    get_device_proc_addr: vk.PfnGetDeviceProcAddr,
    flags: AllocatorCreateFlags = .{},
    preferred_large_heap_block_size: vk.DeviceSize = 0,
    /// Optional per-heap byte limits (length must equal memoryHeapCount).
    heap_size_limit: ?[]const vk.DeviceSize = null,
    /// Optional per-memory-type external handle types (length = memoryTypeCount).
    type_external_memory_handle_types: ?[]const u32 = null,
    allocation_callbacks: ?*const vk.AllocationCallbacks = null,
    device_memory_callbacks: ?*const DeviceMemoryCallbacks = null,
};

pub const AllocationCreateInfo = struct {
    flags: AllocationCreateFlags = .{},
    usage: MemoryUsage = .unknown,
    required_flags: vk.MemoryPropertyFlags = .{},
    preferred_flags: vk.MemoryPropertyFlags = .{},
    memory_type_bits: u32 = 0,
    pool: ?Pool = null,
    user_data: ?*anyopaque = null,
    priority: f32 = 0,

    fn toC(self: AllocationCreateInfo) c.VmaAllocationCreateInfo {
        return .{
            .flags = self.flags.toInt(),
            .usage = @intCast(@intFromEnum(self.usage)),
            .requiredFlags = @bitCast(self.required_flags),
            .preferredFlags = @bitCast(self.preferred_flags),
            .memoryTypeBits = self.memory_type_bits,
            .pool = if (self.pool) |p| p.handle else null,
            .pUserData = self.user_data,
            .priority = self.priority,
        };
    }
};

pub const PoolCreateInfo = struct {
    memory_type_index: u32,
    flags: PoolCreateFlags = .{},
    block_size: vk.DeviceSize = 0,
    min_block_count: usize = 0,
    max_block_count: usize = 0,
    priority: f32 = 0,
    min_allocation_alignment: vk.DeviceSize = 0,
    memory_allocate_next: ?*anyopaque = null,
};

pub const VirtualBlockCreateInfo = struct {
    size: vk.DeviceSize,
    flags: VirtualBlockCreateFlags = .{},
    allocation_callbacks: ?*const vk.AllocationCallbacks = null,
};

pub const VirtualAllocationCreateInfo = struct {
    size: vk.DeviceSize,
    alignment: vk.DeviceSize = 0,
    flags: VirtualAllocationCreateFlags = .{},
    user_data: ?*anyopaque = null,
};

pub const DefragmentationInfo = struct {
    flags: DefragmentationFlags = .{},
    /// Restrict defragmentation to a single custom pool, or `null` for default.
    pool: ?Pool = null,
    max_bytes_per_pass: vk.DeviceSize = 0,
    max_allocations_per_pass: u32 = 0,
    break_callback: ?CheckDefragmentationBreakFn = null,
    break_callback_user_data: ?*anyopaque = null,
};

// ===========================================================================
// Handles
// ===========================================================================

pub const Allocation = struct {
    handle: c.VmaAllocation,
};

pub const Pool = struct {
    handle: c.VmaPool,
};

/// One move proposed by a defragmentation pass. Inspect `src`, record/recreate
/// the data against `dstTmp`, then optionally change the operation (the default
/// is `.copy`).
pub const DefragmentationPass = struct {
    moves: []c.VmaDefragmentationMove,

    pub fn count(self: DefragmentationPass) usize {
        return self.moves.len;
    }
    pub fn src(self: DefragmentationPass, i: usize) Allocation {
        return .{ .handle = self.moves[i].srcAllocation };
    }
    pub fn dstTmp(self: DefragmentationPass, i: usize) Allocation {
        return .{ .handle = self.moves[i].dstTmpAllocation };
    }
    pub fn setOperation(self: DefragmentationPass, i: usize, op: DefragmentationMoveOperation) void {
        self.moves[i].operation = @intCast(@intFromEnum(op));
    }
};

pub const DefragmentationContext = struct {
    handle: c.VmaDefragmentationContext,

    /// Begin a pass. Returns `null` when defragmentation is complete and there is
    /// nothing more to move.
    pub fn beginPass(self: DefragmentationContext, allocator: Allocator) Error!?DefragmentationPass {
        var pass = std.mem.zeroes(c.VmaDefragmentationPassMoveInfo);
        const r = c.vmaBeginDefragmentationPass(allocator.handle, self.handle, &pass);
        if (r == c.VK_SUCCESS) return null;
        if (r != c.VK_INCOMPLETE) try check(r);
        return .{ .moves = pass.pMoves[0..pass.moveCount] };
    }

    /// End a pass. Returns `true` if another pass is needed.
    pub fn endPass(self: DefragmentationContext, allocator: Allocator, pass: DefragmentationPass) Error!bool {
        var info = c.VmaDefragmentationPassMoveInfo{
            .moveCount = @intCast(pass.moves.len),
            .pMoves = pass.moves.ptr,
        };
        const r = c.vmaEndDefragmentationPass(allocator.handle, self.handle, &info);
        if (r == c.VK_INCOMPLETE) return true;
        try check(r);
        return false;
    }
};

pub const VirtualAllocation = struct {
    handle: c.VmaVirtualAllocation,
};

// ===========================================================================
// Allocator
// ===========================================================================

pub const Allocator = struct {
    handle: c.VmaAllocator,

    pub fn create(info: CreateInfo) Error!Allocator {
        var fns = std.mem.zeroes(c.VmaVulkanFunctions);
        fns.vkGetInstanceProcAddr = @ptrCast(info.get_instance_proc_addr);
        fns.vkGetDeviceProcAddr = @ptrCast(info.get_device_proc_addr);

        var mem_cb: c.VmaDeviceMemoryCallbacks = undefined;
        var mem_cb_ptr: [*c]const c.VmaDeviceMemoryCallbacks = null;
        if (info.device_memory_callbacks) |cb| {
            mem_cb = .{
                .pfnAllocate = @ptrCast(cb.allocate),
                .pfnFree = @ptrCast(cb.free),
                .pUserData = cb.user_data,
            };
            mem_cb_ptr = &mem_cb;
        }

        var ci = std.mem.zeroes(c.VmaAllocatorCreateInfo);
        ci.flags = info.flags.toInt();
        ci.instance = toC(c.VkInstance, info.instance);
        ci.physicalDevice = toC(c.VkPhysicalDevice, info.physical_device);
        ci.device = toC(c.VkDevice, info.device);
        ci.vulkanApiVersion = @bitCast(info.api_version);
        ci.pVulkanFunctions = &fns;
        ci.preferredLargeHeapBlockSize = info.preferred_large_heap_block_size;
        ci.pAllocationCallbacks = @ptrCast(info.allocation_callbacks);
        ci.pDeviceMemoryCallbacks = mem_cb_ptr;
        if (info.heap_size_limit) |h| ci.pHeapSizeLimit = h.ptr;
        if (info.type_external_memory_handle_types) |t| ci.pTypeExternalMemoryHandleTypes = t.ptr;

        var handle: c.VmaAllocator = null;
        try check(c.vmaCreateAllocator(&ci, &handle));
        return .{ .handle = handle };
    }

    pub fn destroy(self: Allocator) void {
        c.vmaDestroyAllocator(self.handle);
    }

    // --- introspection -----------------------------------------------------

    pub fn getInfo(self: Allocator) AllocatorInfo {
        var ai: c.VmaAllocatorInfo = undefined;
        c.vmaGetAllocatorInfo(self.handle, &ai);
        return .{
            .instance = toVk(vk.Instance, ai.instance),
            .physical_device = toVk(vk.PhysicalDevice, ai.physicalDevice),
            .device = toVk(vk.Device, ai.device),
        };
    }

    pub fn getPhysicalDeviceProperties(self: Allocator) *const vk.PhysicalDeviceProperties {
        var p: [*c]const c.VkPhysicalDeviceProperties = null;
        c.vmaGetPhysicalDeviceProperties(self.handle, &p);
        return @ptrCast(@alignCast(p));
    }

    pub fn getMemoryProperties(self: Allocator) *const vk.PhysicalDeviceMemoryProperties {
        var p: [*c]const c.VkPhysicalDeviceMemoryProperties = null;
        c.vmaGetMemoryProperties(self.handle, &p);
        return @ptrCast(@alignCast(p));
    }

    pub fn getMemoryTypeProperties(self: Allocator, memory_type_index: u32) vk.MemoryPropertyFlags {
        var flags: c.VkMemoryPropertyFlags = 0;
        c.vmaGetMemoryTypeProperties(self.handle, memory_type_index, &flags);
        return vkFlags(vk.MemoryPropertyFlags, flags);
    }

    /// Number of memory heaps reported by the physical device. Useful to size the
    /// slice passed to `getHeapBudgets`.
    pub fn memoryHeapCount(self: Allocator) u32 {
        return self.getMemoryProperties().memory_heap_count;
    }

    pub fn setCurrentFrameIndex(self: Allocator, frame_index: u32) void {
        c.vmaSetCurrentFrameIndex(self.handle, frame_index);
    }

    pub fn calculateStatistics(self: Allocator) TotalStatistics {
        var stats: c.VmaTotalStatistics = undefined;
        c.vmaCalculateStatistics(self.handle, &stats);
        return TotalStatistics.from(stats);
    }

    /// Fills the first `memoryHeapCount()` entries of `budgets` and returns that
    /// sub-slice. Asserts the buffer is large enough.
    pub fn getHeapBudgets(self: Allocator, budgets: []Budget) []Budget {
        const n = self.memoryHeapCount();
        assert(budgets.len >= n);
        var tmp: [vk.MAX_MEMORY_HEAPS]c.VmaBudget = undefined;
        c.vmaGetHeapBudgets(self.handle, &tmp);
        for (budgets[0..n], tmp[0..n]) |*o, s| o.* = Budget.from(s);
        return budgets[0..n];
    }

    pub fn findMemoryTypeIndex(self: Allocator, memory_type_bits: u32, info: AllocationCreateInfo) Error!u32 {
        var aci = info.toC();
        var out: u32 = 0;
        try check(c.vmaFindMemoryTypeIndex(self.handle, memory_type_bits, &aci, &out));
        return out;
    }

    pub fn findMemoryTypeIndexForBufferInfo(self: Allocator, buffer_info: *const vk.BufferCreateInfo, info: AllocationCreateInfo) Error!u32 {
        var aci = info.toC();
        var out: u32 = 0;
        try check(c.vmaFindMemoryTypeIndexForBufferInfo(self.handle, @ptrCast(buffer_info), &aci, &out));
        return out;
    }

    pub fn findMemoryTypeIndexForImageInfo(self: Allocator, image_info: *const vk.ImageCreateInfo, info: AllocationCreateInfo) Error!u32 {
        var aci = info.toC();
        var out: u32 = 0;
        try check(c.vmaFindMemoryTypeIndexForImageInfo(self.handle, @ptrCast(image_info), &aci, &out));
        return out;
    }

    pub fn checkCorruption(self: Allocator, memory_type_bits: u32) Error!void {
        try check(c.vmaCheckCorruption(self.handle, memory_type_bits));
    }

    // --- custom pools ------------------------------------------------------

    pub fn createPool(self: Allocator, info: PoolCreateInfo) Error!Pool {
        const ci = c.VmaPoolCreateInfo{
            .memoryTypeIndex = info.memory_type_index,
            .flags = info.flags.toInt(),
            .blockSize = info.block_size,
            .minBlockCount = info.min_block_count,
            .maxBlockCount = info.max_block_count,
            .priority = info.priority,
            .minAllocationAlignment = info.min_allocation_alignment,
            .pMemoryAllocateNext = info.memory_allocate_next,
        };
        var pool: c.VmaPool = null;
        try check(c.vmaCreatePool(self.handle, &ci, &pool));
        return .{ .handle = pool };
    }

    pub fn destroyPool(self: Allocator, pool: Pool) void {
        c.vmaDestroyPool(self.handle, pool.handle);
    }

    pub fn getPoolStatistics(self: Allocator, pool: Pool) Statistics {
        var s: c.VmaStatistics = undefined;
        c.vmaGetPoolStatistics(self.handle, pool.handle, &s);
        return Statistics.from(s);
    }

    pub fn calculatePoolStatistics(self: Allocator, pool: Pool) DetailedStatistics {
        var s: c.VmaDetailedStatistics = undefined;
        c.vmaCalculatePoolStatistics(self.handle, pool.handle, &s);
        return DetailedStatistics.from(s);
    }

    pub fn checkPoolCorruption(self: Allocator, pool: Pool) Error!void {
        try check(c.vmaCheckPoolCorruption(self.handle, pool.handle));
    }

    pub fn getPoolName(self: Allocator, pool: Pool) ?[*:0]const u8 {
        var name: [*c]const u8 = null;
        c.vmaGetPoolName(self.handle, pool.handle, &name);
        return optName(name);
    }

    pub fn setPoolName(self: Allocator, pool: Pool, name: ?[*:0]const u8) void {
        c.vmaSetPoolName(self.handle, pool.handle, name orelse null);
    }

    // --- explicit allocation ----------------------------------------------

    pub fn allocateMemory(self: Allocator, requirements: *const vk.MemoryRequirements, info: AllocationCreateInfo) Error!Allocation {
        var aci = info.toC();
        var alloc: c.VmaAllocation = null;
        try check(c.vmaAllocateMemory(self.handle, @ptrCast(requirements), &aci, &alloc, null));
        return .{ .handle = alloc };
    }

    /// `out_allocations.len` is the number of pages to allocate; each receives
    /// one allocation with the same requirements.
    pub fn allocateMemoryPages(self: Allocator, requirements: *const vk.MemoryRequirements, info: AllocationCreateInfo, out_allocations: []Allocation) Error!void {
        var aci = info.toC();
        try check(c.vmaAllocateMemoryPages(
            self.handle,
            @ptrCast(requirements),
            &aci,
            out_allocations.len,
            @ptrCast(out_allocations.ptr),
            null,
        ));
    }

    pub fn allocateMemoryForBuffer(self: Allocator, buffer: vk.Buffer, info: AllocationCreateInfo) Error!Allocation {
        var aci = info.toC();
        var alloc: c.VmaAllocation = null;
        try check(c.vmaAllocateMemoryForBuffer(self.handle, toC(c.VkBuffer, buffer), &aci, &alloc, null));
        return .{ .handle = alloc };
    }

    pub fn allocateMemoryForImage(self: Allocator, image: vk.Image, info: AllocationCreateInfo) Error!Allocation {
        var aci = info.toC();
        var alloc: c.VmaAllocation = null;
        try check(c.vmaAllocateMemoryForImage(self.handle, toC(c.VkImage, image), &aci, &alloc, null));
        return .{ .handle = alloc };
    }

    pub fn freeMemory(self: Allocator, allocation: Allocation) void {
        c.vmaFreeMemory(self.handle, allocation.handle);
    }

    pub fn freeMemoryPages(self: Allocator, allocations: []const Allocation) void {
        c.vmaFreeMemoryPages(self.handle, allocations.len, @ptrCast(allocations.ptr));
    }

    // --- allocation info & user data --------------------------------------

    pub fn getAllocationInfo(self: Allocator, allocation: Allocation) AllocationInfo {
        var ci: c.VmaAllocationInfo = undefined;
        c.vmaGetAllocationInfo(self.handle, allocation.handle, &ci);
        return AllocationInfo.from(ci);
    }

    pub fn getAllocationInfo2(self: Allocator, allocation: Allocation) AllocationInfo2 {
        var ci: c.VmaAllocationInfo2 = undefined;
        c.vmaGetAllocationInfo2(self.handle, allocation.handle, &ci);
        return .{
            .info = AllocationInfo.from(ci.allocationInfo),
            .block_size = ci.blockSize,
            .dedicated_memory = ci.dedicatedMemory != 0,
        };
    }

    pub fn setAllocationUserData(self: Allocator, allocation: Allocation, user_data: ?*anyopaque) void {
        c.vmaSetAllocationUserData(self.handle, allocation.handle, user_data);
    }

    pub fn setAllocationName(self: Allocator, allocation: Allocation, name: ?[*:0]const u8) void {
        c.vmaSetAllocationName(self.handle, allocation.handle, name orelse null);
    }

    pub fn getAllocationMemoryProperties(self: Allocator, allocation: Allocation) vk.MemoryPropertyFlags {
        var flags: c.VkMemoryPropertyFlags = 0;
        c.vmaGetAllocationMemoryProperties(self.handle, allocation.handle, &flags);
        return vkFlags(vk.MemoryPropertyFlags, flags);
    }

    // --- mapping & coherency ----------------------------------------------

    /// Map and return the start of the allocation's host-visible memory.
    pub fn mapMemory(self: Allocator, allocation: Allocation) Error![*]u8 {
        var ptr: ?*anyopaque = null;
        try check(c.vmaMapMemory(self.handle, allocation.handle, &ptr));
        return @ptrCast(ptr.?);
    }

    /// Map and reinterpret the memory as a many-item pointer of `T`.
    pub fn mapMemoryAs(self: Allocator, comptime T: type, allocation: Allocation) Error![*]T {
        var ptr: ?*anyopaque = null;
        try check(c.vmaMapMemory(self.handle, allocation.handle, &ptr));
        return @ptrCast(@alignCast(ptr.?));
    }

    pub fn unmapMemory(self: Allocator, allocation: Allocation) void {
        c.vmaUnmapMemory(self.handle, allocation.handle);
    }

    pub fn flushAllocation(self: Allocator, allocation: Allocation, offset: vk.DeviceSize, size: vk.DeviceSize) Error!void {
        try check(c.vmaFlushAllocation(self.handle, allocation.handle, offset, size));
    }

    pub fn invalidateAllocation(self: Allocator, allocation: Allocation, offset: vk.DeviceSize, size: vk.DeviceSize) Error!void {
        try check(c.vmaInvalidateAllocation(self.handle, allocation.handle, offset, size));
    }

    pub fn flushAllocations(self: Allocator, allocations: []const Allocation, offsets: ?[]const vk.DeviceSize, sizes: ?[]const vk.DeviceSize) Error!void {
        try check(c.vmaFlushAllocations(
            self.handle,
            @intCast(allocations.len),
            @ptrCast(allocations.ptr),
            if (offsets) |o| o.ptr else null,
            if (sizes) |s| s.ptr else null,
        ));
    }

    pub fn invalidateAllocations(self: Allocator, allocations: []const Allocation, offsets: ?[]const vk.DeviceSize, sizes: ?[]const vk.DeviceSize) Error!void {
        try check(c.vmaInvalidateAllocations(
            self.handle,
            @intCast(allocations.len),
            @ptrCast(allocations.ptr),
            if (offsets) |o| o.ptr else null,
            if (sizes) |s| s.ptr else null,
        ));
    }

    pub fn copyMemoryToAllocation(self: Allocator, src: []const u8, dst: Allocation, dst_offset: vk.DeviceSize) Error!void {
        try check(c.vmaCopyMemoryToAllocation(self.handle, src.ptr, dst.handle, dst_offset, src.len));
    }

    pub fn copyAllocationToMemory(self: Allocator, src: Allocation, src_offset: vk.DeviceSize, dst: []u8) Error!void {
        try check(c.vmaCopyAllocationToMemory(self.handle, src.handle, src_offset, dst.ptr, dst.len));
    }

    // --- binding -----------------------------------------------------------

    pub fn bindBufferMemory(self: Allocator, allocation: Allocation, buffer: vk.Buffer) Error!void {
        try check(c.vmaBindBufferMemory(self.handle, allocation.handle, toC(c.VkBuffer, buffer)));
    }

    pub fn bindBufferMemory2(self: Allocator, allocation: Allocation, local_offset: vk.DeviceSize, buffer: vk.Buffer, next: ?*const anyopaque) Error!void {
        try check(c.vmaBindBufferMemory2(self.handle, allocation.handle, local_offset, toC(c.VkBuffer, buffer), next));
    }

    pub fn bindImageMemory(self: Allocator, allocation: Allocation, image: vk.Image) Error!void {
        try check(c.vmaBindImageMemory(self.handle, allocation.handle, toC(c.VkImage, image)));
    }

    pub fn bindImageMemory2(self: Allocator, allocation: Allocation, local_offset: vk.DeviceSize, image: vk.Image, next: ?*const anyopaque) Error!void {
        try check(c.vmaBindImageMemory2(self.handle, allocation.handle, local_offset, toC(c.VkImage, image), next));
    }

    // --- resource creation -------------------------------------------------

    pub const BufferResult = struct {
        buffer: vk.Buffer,
        allocation: Allocation,
        info: AllocationInfo,
    };

    pub const ImageResult = struct {
        image: vk.Image,
        allocation: Allocation,
        info: AllocationInfo,
    };

    pub fn createBuffer(self: Allocator, buffer_info: *const vk.BufferCreateInfo, alloc_info: AllocationCreateInfo) Error!BufferResult {
        var aci = alloc_info.toC();
        var buffer: c.VkBuffer = null;
        var alloc: c.VmaAllocation = null;
        var out: c.VmaAllocationInfo = undefined;
        try check(c.vmaCreateBuffer(self.handle, @ptrCast(buffer_info), &aci, &buffer, &alloc, &out));
        return .{ .buffer = toVk(vk.Buffer, buffer), .allocation = .{ .handle = alloc }, .info = AllocationInfo.from(out) };
    }

    pub fn createBufferWithAlignment(self: Allocator, buffer_info: *const vk.BufferCreateInfo, alloc_info: AllocationCreateInfo, min_alignment: vk.DeviceSize) Error!BufferResult {
        var aci = alloc_info.toC();
        var buffer: c.VkBuffer = null;
        var alloc: c.VmaAllocation = null;
        var out: c.VmaAllocationInfo = undefined;
        try check(c.vmaCreateBufferWithAlignment(self.handle, @ptrCast(buffer_info), &aci, min_alignment, &buffer, &alloc, &out));
        return .{ .buffer = toVk(vk.Buffer, buffer), .allocation = .{ .handle = alloc }, .info = AllocationInfo.from(out) };
    }

    /// Create a buffer aliasing the memory of an existing allocation.
    pub fn createAliasingBuffer(self: Allocator, allocation: Allocation, buffer_info: *const vk.BufferCreateInfo) Error!vk.Buffer {
        var buffer: c.VkBuffer = null;
        try check(c.vmaCreateAliasingBuffer(self.handle, allocation.handle, @ptrCast(buffer_info), &buffer));
        return toVk(vk.Buffer, buffer);
    }

    pub fn createAliasingBuffer2(self: Allocator, allocation: Allocation, local_offset: vk.DeviceSize, buffer_info: *const vk.BufferCreateInfo) Error!vk.Buffer {
        var buffer: c.VkBuffer = null;
        try check(c.vmaCreateAliasingBuffer2(self.handle, allocation.handle, local_offset, @ptrCast(buffer_info), &buffer));
        return toVk(vk.Buffer, buffer);
    }

    pub fn destroyBuffer(self: Allocator, buffer: vk.Buffer, allocation: Allocation) void {
        c.vmaDestroyBuffer(self.handle, toC(c.VkBuffer, buffer), allocation.handle);
    }

    pub fn createImage(self: Allocator, image_info: *const vk.ImageCreateInfo, alloc_info: AllocationCreateInfo) Error!ImageResult {
        var aci = alloc_info.toC();
        var image: c.VkImage = null;
        var alloc: c.VmaAllocation = null;
        var out: c.VmaAllocationInfo = undefined;
        try check(c.vmaCreateImage(self.handle, @ptrCast(image_info), &aci, &image, &alloc, &out));
        return .{ .image = toVk(vk.Image, image), .allocation = .{ .handle = alloc }, .info = AllocationInfo.from(out) };
    }

    pub fn createAliasingImage(self: Allocator, allocation: Allocation, image_info: *const vk.ImageCreateInfo) Error!vk.Image {
        var image: c.VkImage = null;
        try check(c.vmaCreateAliasingImage(self.handle, allocation.handle, @ptrCast(image_info), &image));
        return toVk(vk.Image, image);
    }

    pub fn createAliasingImage2(self: Allocator, allocation: Allocation, local_offset: vk.DeviceSize, image_info: *const vk.ImageCreateInfo) Error!vk.Image {
        var image: c.VkImage = null;
        try check(c.vmaCreateAliasingImage2(self.handle, allocation.handle, local_offset, @ptrCast(image_info), &image));
        return toVk(vk.Image, image);
    }

    pub fn destroyImage(self: Allocator, image: vk.Image, allocation: Allocation) void {
        c.vmaDestroyImage(self.handle, toC(c.VkImage, image), allocation.handle);
    }

    // --- defragmentation ---------------------------------------------------

    pub fn beginDefragmentation(self: Allocator, info: DefragmentationInfo) Error!DefragmentationContext {
        const ci = c.VmaDefragmentationInfo{
            .flags = info.flags.toInt(),
            .pool = if (info.pool) |p| p.handle else null,
            .maxBytesPerPass = info.max_bytes_per_pass,
            .maxAllocationsPerPass = info.max_allocations_per_pass,
            .pfnBreakCallback = @ptrCast(info.break_callback),
            .pBreakCallbackUserData = info.break_callback_user_data,
        };
        var ctx: c.VmaDefragmentationContext = null;
        try check(c.vmaBeginDefragmentation(self.handle, &ci, &ctx));
        return .{ .handle = ctx };
    }

    pub fn endDefragmentation(self: Allocator, ctx: DefragmentationContext) DefragmentationStats {
        var stats: c.VmaDefragmentationStats = undefined;
        c.vmaEndDefragmentation(self.handle, ctx.handle, &stats);
        return .{
            .bytes_moved = stats.bytesMoved,
            .bytes_freed = stats.bytesFreed,
            .allocations_moved = stats.allocationsMoved,
            .device_memory_blocks_freed = stats.deviceMemoryBlocksFreed,
        };
    }

    // --- JSON dump ---------------------------------------------------------

    /// Caller must free the result with `freeStatsString`.
    pub fn buildStatsString(self: Allocator, detailed: bool) [:0]u8 {
        var s: [*c]u8 = null;
        c.vmaBuildStatsString(self.handle, &s, @intFromBool(detailed));
        return std.mem.span(@as([*:0]u8, @ptrCast(s)));
    }

    pub fn freeStatsString(self: Allocator, s: [:0]u8) void {
        c.vmaFreeStatsString(self.handle, s.ptr);
    }
};

// ===========================================================================
// Virtual allocator (uses VMA's algorithm without any VkDeviceMemory)
// ===========================================================================

pub const VirtualBlock = struct {
    handle: c.VmaVirtualBlock,

    pub fn create(info: VirtualBlockCreateInfo) Error!VirtualBlock {
        const ci = c.VmaVirtualBlockCreateInfo{
            .size = info.size,
            .flags = info.flags.toInt(),
            .pAllocationCallbacks = @ptrCast(info.allocation_callbacks),
        };
        var block: c.VmaVirtualBlock = null;
        try check(c.vmaCreateVirtualBlock(&ci, &block));
        return .{ .handle = block };
    }

    pub fn destroy(self: VirtualBlock) void {
        c.vmaDestroyVirtualBlock(self.handle);
    }

    pub fn isEmpty(self: VirtualBlock) bool {
        return c.vmaIsVirtualBlockEmpty(self.handle) != 0;
    }

    pub const AllocResult = struct { allocation: VirtualAllocation, offset: vk.DeviceSize };

    pub fn allocate(self: VirtualBlock, info: VirtualAllocationCreateInfo) Error!AllocResult {
        const ci = c.VmaVirtualAllocationCreateInfo{
            .size = info.size,
            .alignment = info.alignment,
            .flags = info.flags.toInt(),
            .pUserData = info.user_data,
        };
        var alloc: c.VmaVirtualAllocation = null;
        var offset: vk.DeviceSize = 0;
        try check(c.vmaVirtualAllocate(self.handle, &ci, &alloc, &offset));
        return .{ .allocation = .{ .handle = alloc }, .offset = offset };
    }

    pub fn free(self: VirtualBlock, allocation: VirtualAllocation) void {
        c.vmaVirtualFree(self.handle, allocation.handle);
    }

    pub fn clear(self: VirtualBlock) void {
        c.vmaClearVirtualBlock(self.handle);
    }

    pub fn getAllocationInfo(self: VirtualBlock, allocation: VirtualAllocation) VirtualAllocationInfo {
        var ci: c.VmaVirtualAllocationInfo = undefined;
        c.vmaGetVirtualAllocationInfo(self.handle, allocation.handle, &ci);
        return .{ .offset = ci.offset, .size = ci.size, .user_data = ci.pUserData };
    }

    pub fn setAllocationUserData(self: VirtualBlock, allocation: VirtualAllocation, user_data: ?*anyopaque) void {
        c.vmaSetVirtualAllocationUserData(self.handle, allocation.handle, user_data);
    }

    pub fn getStatistics(self: VirtualBlock) Statistics {
        var s: c.VmaStatistics = undefined;
        c.vmaGetVirtualBlockStatistics(self.handle, &s);
        return Statistics.from(s);
    }

    pub fn calculateStatistics(self: VirtualBlock) DetailedStatistics {
        var s: c.VmaDetailedStatistics = undefined;
        c.vmaCalculateVirtualBlockStatistics(self.handle, &s);
        return DetailedStatistics.from(s);
    }

    /// Caller must free the result with `freeStatsString`.
    pub fn buildStatsString(self: VirtualBlock, detailed: bool) [:0]u8 {
        var s: [*c]u8 = null;
        c.vmaBuildVirtualBlockStatsString(self.handle, &s, @intFromBool(detailed));
        return std.mem.span(@as([*:0]u8, @ptrCast(s)));
    }

    pub fn freeStatsString(self: VirtualBlock, s: [:0]u8) void {
        c.vmaFreeVirtualBlockStatsString(self.handle, s.ptr);
    }
};

// ===========================================================================
// Compile-time verification
// ===========================================================================
//
// These never run; they exist so the build fails loudly if the hand-maintained
// flag bit positions or the re-interpreted struct layouts ever diverge from the
// VMA headers that were actually compiled.

fn bitOf(comptime Flags: type, comptime field: []const u8) u32 {
    var v = Flags{};
    @field(v, field) = true;
    return v.toInt();
}

comptime {
    // --- Flag bit positions -------------------------------------------------
    assert(bitOf(AllocatorCreateFlags, "externally_synchronized") == c.VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT);
    assert(bitOf(AllocatorCreateFlags, "khr_dedicated_allocation") == c.VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT);
    assert(bitOf(AllocatorCreateFlags, "khr_bind_memory2") == c.VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT);
    assert(bitOf(AllocatorCreateFlags, "ext_memory_budget") == c.VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT);
    assert(bitOf(AllocatorCreateFlags, "amd_device_coherent_memory") == c.VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT);
    assert(bitOf(AllocatorCreateFlags, "buffer_device_address") == c.VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
    assert(bitOf(AllocatorCreateFlags, "ext_memory_priority") == c.VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT);
    assert(bitOf(AllocatorCreateFlags, "khr_maintenance4") == c.VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT);
    assert(bitOf(AllocatorCreateFlags, "khr_maintenance5") == c.VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT);
    assert(bitOf(AllocatorCreateFlags, "khr_external_memory_win32") == c.VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT);

    assert(bitOf(AllocationCreateFlags, "dedicated_memory") == c.VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(bitOf(AllocationCreateFlags, "never_allocate") == c.VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT);
    assert(bitOf(AllocationCreateFlags, "mapped") == c.VMA_ALLOCATION_CREATE_MAPPED_BIT);
    assert(bitOf(AllocationCreateFlags, "user_data_copy_string") == c.VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT);
    assert(bitOf(AllocationCreateFlags, "upper_address") == c.VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT);
    assert(bitOf(AllocationCreateFlags, "dont_bind") == c.VMA_ALLOCATION_CREATE_DONT_BIND_BIT);
    assert(bitOf(AllocationCreateFlags, "within_budget") == c.VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
    assert(bitOf(AllocationCreateFlags, "can_alias") == c.VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT);
    assert(bitOf(AllocationCreateFlags, "host_access_sequential_write") == c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    assert(bitOf(AllocationCreateFlags, "host_access_random") == c.VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    assert(bitOf(AllocationCreateFlags, "host_access_allow_transfer_instead") == c.VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    assert(bitOf(AllocationCreateFlags, "strategy_min_memory") == c.VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT);
    assert(bitOf(AllocationCreateFlags, "strategy_min_time") == c.VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT);
    assert(bitOf(AllocationCreateFlags, "strategy_min_offset") == c.VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT);

    assert(bitOf(PoolCreateFlags, "ignore_buffer_image_granularity") == c.VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT);
    assert(bitOf(PoolCreateFlags, "linear_algorithm") == c.VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT);

    assert(bitOf(DefragmentationFlags, "algorithm_fast") == c.VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT);
    assert(bitOf(DefragmentationFlags, "algorithm_balanced") == c.VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT);
    assert(bitOf(DefragmentationFlags, "algorithm_full") == c.VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT);
    assert(bitOf(DefragmentationFlags, "algorithm_extensive") == c.VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT);

    assert(bitOf(VirtualBlockCreateFlags, "linear_algorithm") == c.VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT);

    assert(bitOf(VirtualAllocationCreateFlags, "upper_address") == c.VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT);
    assert(bitOf(VirtualAllocationCreateFlags, "strategy_min_memory") == c.VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT);
    assert(bitOf(VirtualAllocationCreateFlags, "strategy_min_time") == c.VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT);
    assert(bitOf(VirtualAllocationCreateFlags, "strategy_min_offset") == c.VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT);

    // --- Re-interpreted struct sizes ---------------------------------------
    //
    // These vk.* structs are passed straight through to the C API by pointer
    // cast, so their layout must match byte-for-byte.
    assert(@sizeOf(vk.MemoryRequirements) == @sizeOf(c.VkMemoryRequirements));
    assert(@sizeOf(vk.BufferCreateInfo) == @sizeOf(c.VkBufferCreateInfo));
    assert(@sizeOf(vk.ImageCreateInfo) == @sizeOf(c.VkImageCreateInfo));
    assert(@sizeOf(vk.PhysicalDeviceProperties) == @sizeOf(c.VkPhysicalDeviceProperties));
    assert(@sizeOf(vk.PhysicalDeviceMemoryProperties) == @sizeOf(c.VkPhysicalDeviceMemoryProperties));
    assert(@sizeOf(vk.AllocationCallbacks) == @sizeOf(c.VkAllocationCallbacks));
    assert(@bitSizeOf(vk.MemoryPropertyFlags) == @bitSizeOf(c.VkMemoryPropertyFlags));

    // --- Statistics array bounds match VMA's expectations ------------------
    assert(vk.MAX_MEMORY_TYPES == @typeInfo(@FieldType(c.VmaTotalStatistics, "memoryType")).array.len);
    assert(vk.MAX_MEMORY_HEAPS == @typeInfo(@FieldType(c.VmaTotalStatistics, "memoryHeap")).array.len);
}

fn refAllDeclsRecursive(comptime T: type) void {
    inline for (comptime std.meta.declarations(T)) |decl| {
        const field = @field(T, decl.name);
        if (@TypeOf(field) == type) {
            switch (@typeInfo(field)) {
                .@"struct", .@"enum", .@"union", .@"opaque" => refAllDeclsRecursive(field),
                else => {},
            }
        }
        _ = &field;
    }
}

test "compile-time layout verification is reachable" {
    // Forces every method body to be analyzed; the real checks are the `comptime`
    // block above.
    refAllDeclsRecursive(@This());
}
