const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Neither upstream package ships a build.zig, so we only ask them for paths.
    const vma_dep = b.dependency("VulkanMemoryAllocator", .{});
    const vk_headers_dep = b.dependency("vulkan_headers", .{});

    const vma_include = vma_dep.path("include");
    const vk_include = vk_headers_dep.path("include");

    // The private C view of VMA, imported by root.zig as "vma_h". Declares extern
    // functions only; the symbols come from the static library below.
    const translate_c = b.addTranslateC(.{
        .root_source_file = vma_dep.path("include/vk_mem_alloc.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.defineCMacro("VK_NO_PROTOTYPES", "1");
    translate_c.addIncludePath(vma_include);
    translate_c.addIncludePath(vk_include);
    const c_mod = translate_c.createModule();

    // VMA_IMPLEMENTATION compiled exactly once, here; consumers must never
    // compile it themselves. The dynamic-functions path makes VMA load Vulkan
    // entry points through the proc-addr pointers passed at allocator creation,
    // so the consumer needn't link Vulkan statically.
    const vma_impl = b.addWriteFiles().add("vma_impl.cpp",
        \\#define VK_NO_PROTOTYPES
        \\#define VMA_STATIC_VULKAN_FUNCTIONS 0
        \\#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
        \\#define VMA_IMPLEMENTATION
        \\#include "vk_mem_alloc.h"
        \\
    );

    const vma_lib = b.addLibrary(.{
        .name = "vma",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    vma_lib.root_module.addCSourceFile(.{
        .file = vma_impl,
        .flags = &.{ "-std=c++17", "-fno-exceptions", "-fno-rtti" },
    });
    vma_lib.root_module.addIncludePath(vma_include);
    vma_lib.root_module.addIncludePath(vk_include);
    vma_lib.root_module.link_libc = true;
    vma_lib.root_module.link_libcpp = true;
    b.installArtifact(vma_lib);

    // The public module. The implementation library travels with it; the
    // consumer adds their own "vulkan" import so there is exactly one vulkan-zig
    // in their dependency graph.
    const vma_mod = b.addModule("vma", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    vma_mod.addImport("vma_h", c_mod);
    vma_mod.linkLibrary(vma_lib);

    // Dev-only vulkan-zig, wired into a private copy of the module so the public
    // one stays free of a vulkan import that could clash with the consumer's.
    const vulkan_dep = b.lazyDependency("vulkan_zig", .{
        .registry = vk_headers_dep.path("registry/vk.xml"),
    }) orelse return; // not fetched (e.g. downstream build); skip dev-only steps
    const vulkan_mod = vulkan_dep.module("vulkan-zig");

    const vma_dev = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    vma_dev.addImport("vma_h", c_mod);
    vma_dev.linkLibrary(vma_lib);
    vma_dev.addImport("vulkan", vulkan_mod);

    const tests = b.addTest(.{ .root_module = vma_dev });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the binding's tests");
    test_step.dependOn(&run_tests.step);

    // Examples. Each is built by `zig build examples`; the virtual-allocator one
    // needs no GPU and can be run with `zig build run-virtual`.
    const examples_step = b.step("examples", "Build the examples");
    for ([_][]const u8{ "virtual_block", "buffers" }) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "vma", .module = vma_dev },
                    .{ .name = "vulkan", .module = vulkan_mod },
                },
            }),
        });
        examples_step.dependOn(&exe.step);
        b.installArtifact(exe);

        if (std.mem.eql(u8, name, "virtual_block")) {
            const run = b.addRunArtifact(exe);
            b.step("run-virtual", "Run the virtual-allocator example").dependOn(&run.step);
        }
    }
}
