//! The command wrappers' shape rules and protocol forms, and the runtime's create commands, on
//! mock function tables built with `from_parts`: each mock records what it received in
//! thread-local state (tests run in parallel threads), and the tests assert on the records. A mock
//! never asserts: a panic cannot unwind out of an `extern "system"` function, so it would abort
//! every test in the binary.

use core::cell::{Cell, RefCell};
use core::ffi::{CStr, c_char, c_void};
use core::ptr;
use std::panic::{self, AssertUnwindSafe};
use tessera_vk::{Device, DeviceTable, Entry, EntryTable, Instance, InstanceTable, vk};

thread_local! {
    /// Calls the current mock received.
    static CALLS: Cell<u32> = const { Cell::new(0) };
    /// Values the current mock recorded.
    static SEEN: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    /// Records a consuming mock still holds, oldest first.
    static RECORDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Calls the current mock received.
fn calls() -> u32 {
    CALLS.with(Cell::get)
}

/// Counts a call; returns its number (1-based).
fn call() -> u32 {
    CALLS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}

/// Records `v`.
fn see(v: u64) {
    SEEN.with(|s| s.borrow_mut().push(v));
}

/// The values recorded so far.
fn seen() -> Vec<u64> {
    SEEN.with(|s| s.borrow().clone())
}

/// A device with handle `0x1000` over `table`.
fn device(table: DeviceTable) -> Device {
    Device::from_parts(vk::Device::from_raw(0x1000), table)
}

/// Rule 1: count query, fill, and a repeat of both on `INCOMPLETE`; the result is truncated.
/// Records the device handle on every call and the capacity each fill passes.
unsafe extern "system" fn swapchain_images(
    device: vk::Device,
    _: vk::SwapchainKHR,
    count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    see(device.as_raw() as u64);
    let n = call();
    // SAFETY: the wrapper passes a live count and, when non-null, an array of `*count` images.
    unsafe {
        if images.is_null() {
            *count = 3;
            return vk::Result::SUCCESS;
        }
        see(u64::from(*count));
        if n == 2 {
            return vk::Result::INCOMPLETE;
        }
        if *count < 2 {
            return vk::Result::ERROR_UNKNOWN;
        }
        *count = 2;
        *images = vk::Image::from_raw(7);
        *images.add(1) = vk::Image::from_raw(8);
    }
    vk::Result::SUCCESS
}

#[test]
fn size_then_fill_repeats_on_incomplete_and_truncates() {
    let d = device(DeviceTable {
        get_swapchain_images_khr: Some(swapchain_images),
        ..Default::default()
    });
    // SAFETY: the mock reads and writes only what the wrapper passes.
    let images = unsafe { d.get_swapchain_images_khr(vk::SwapchainKHR::from_raw(1)) };
    assert_eq!(
        images,
        Ok(vec![vk::Image::from_raw(7), vk::Image::from_raw(8)])
    );
    assert_eq!(calls(), 4, "query, fill (INCOMPLETE), query, fill");
    assert_eq!(
        seen(),
        [0x1000, 0x1000, 3, 0x1000, 0x1000, 3],
        "the device handle on every call; each fill passes the queried capacity"
    );
}

/// Rule 1 with an output before the size/data pair (`vkGetPipelineBinaryDataKHR`): the key and
/// every other argument reach both calls unchanged. Records 1 on each call that gets both the
/// info and the key (0 otherwise), and the size the fill passes.
unsafe extern "system" fn binary_data(
    _: vk::Device,
    info: *const vk::PipelineBinaryDataInfoKHR<'_>,
    key: *mut vk::PipelineBinaryKeyKHR<'_>,
    size: *mut usize,
    data: *mut c_void,
) -> vk::Result {
    call();
    let given = !info.is_null() && !key.is_null();
    see(u64::from(given));
    if !given {
        return vk::Result::ERROR_UNKNOWN;
    }
    // SAFETY: the wrapper passes live pointers; `data` holds `*size` bytes when non-null.
    unsafe {
        (*key).key_size = 4;
        if data.is_null() {
            *size = 5;
            return vk::Result::SUCCESS;
        }
        see(*size as u64);
        if *size != 5 {
            return vk::Result::ERROR_UNKNOWN;
        }
        ptr::copy_nonoverlapping([1u8, 2, 3, 4, 5].as_ptr(), data.cast(), 5);
    }
    vk::Result::SUCCESS
}

#[test]
fn size_then_fill_passes_other_arguments_to_both_calls() {
    let d = device(DeviceTable {
        get_pipeline_binary_data_khr: Some(binary_data),
        ..Default::default()
    });
    let info = vk::PipelineBinaryDataInfoKHR::default();
    let mut key = vk::PipelineBinaryKeyKHR::default();
    // SAFETY: the mock reads and writes only what the wrapper passes.
    let data = unsafe { d.get_pipeline_binary_data_khr(&info, &mut key) };
    assert_eq!(data, Ok(vec![1, 2, 3, 4, 5]));
    assert_eq!(key.key_size, 4);
    assert_eq!(calls(), 2);
    assert_eq!(
        seen(),
        [1, 1, 5],
        "info and key on both calls; the fill passes the queried size, not zero"
    );
}

/// Rule 2 (a trailing output without `sType` returned by value) with rule 3 (`SUBOPTIMAL_KHR`
/// returned too).
unsafe extern "system" fn acquire(
    _: vk::Device,
    _: vk::SwapchainKHR,
    timeout: u64,
    _: vk::Semaphore,
    _: vk::Fence,
    index: *mut u32,
) -> vk::Result {
    see(timeout);
    // SAFETY: the wrapper passes a live index.
    unsafe { *index = 2 };
    vk::Result::SUBOPTIMAL_KHR
}

/// `vkGetBufferMemoryRequirements` writing a size of 256.
unsafe extern "system" fn requirements(
    _: vk::Device,
    _: vk::Buffer,
    out: *mut vk::MemoryRequirements,
) {
    // SAFETY: the wrapper passes a live struct.
    unsafe { (*out).size = 256 };
}

#[test]
fn trailing_outputs_are_returned_and_success_codes_kept() {
    let d = device(DeviceTable {
        acquire_next_image_khr: Some(acquire),
        get_buffer_memory_requirements: Some(requirements),
        ..Default::default()
    });
    // SAFETY: the mocks write only through the pointers the wrappers pass.
    unsafe {
        let acquired = d.acquire_next_image_khr(
            vk::SwapchainKHR::from_raw(1),
            99,
            vk::Semaphore::null(),
            vk::Fence::null(),
        );
        assert_eq!(acquired, Ok((2, vk::Result::SUBOPTIMAL_KHR)));
        assert_eq!(seen(), [99]);
        let req = d.get_buffer_memory_requirements(vk::Buffer::from_raw(1));
        assert_eq!(req.size, 256);
    }
}

/// `vkWaitForFences` recording the count, the second fence and `waitAll`; returns `TIMEOUT`.
unsafe extern "system" fn wait(
    _: vk::Device,
    count: u32,
    fences: *const vk::Fence,
    wait_all: vk::Bool32,
    _: u64,
) -> vk::Result {
    see(u64::from(count));
    // SAFETY: the wrapper passes `count` fences.
    see(unsafe { (*fences.add(1)).as_raw() });
    see(u64::from(wait_all));
    vk::Result::TIMEOUT
}

/// `vkGetFenceStatus` returning `NOT_READY`.
unsafe extern "system" fn fence_status(_: vk::Device, _: vk::Fence) -> vk::Result {
    vk::Result::NOT_READY
}

/// `vkGetFenceStatus` returning `ERROR_DEVICE_LOST`.
unsafe extern "system" fn lost(_: vk::Device, _: vk::Fence) -> vk::Result {
    vk::Result::ERROR_DEVICE_LOST
}

/// `vkCreateBuffer` returning `NOT_READY`, a code its registry entry does not list.
unsafe extern "system" fn create_buffer_positive(
    _: vk::Device,
    _: *const vk::BufferCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    _: *mut vk::Buffer,
) -> vk::Result {
    vk::Result::NOT_READY
}

#[test]
fn success_codes_are_returned_and_errors_are_err() {
    let fences = [vk::Fence::from_raw(1), vk::Fence::from_raw(2)];
    let d = device(DeviceTable {
        wait_for_fences: Some(wait),
        get_fence_status: Some(fence_status),
        create_buffer: Some(create_buffer_positive),
        ..Default::default()
    });
    // SAFETY: the mocks read only what the wrappers pass.
    unsafe {
        assert_eq!(d.wait_for_fences(&fences, true, 5), Ok(vk::Result::TIMEOUT));
        assert_eq!(
            seen(),
            [2, 2, 1],
            "the count is the slice length; true is VK_TRUE"
        );
        assert_eq!(d.get_fence_status(fences[0]), Ok(vk::Result::NOT_READY));
        assert_eq!(
            d.create_buffer(&vk::BufferCreateInfo::default(), None),
            Err(vk::Result::NOT_READY),
            "a SUCCESS-only command reports any other code as an error"
        );
    }
    let d = device(DeviceTable {
        get_fence_status: Some(lost),
        ..Default::default()
    });
    // SAFETY: the mock takes nothing by pointer.
    let status = unsafe { d.get_fence_status(fences[0]) };
    assert_eq!(status, Err(vk::Result::ERROR_DEVICE_LOST));
}

/// `vkCmdBindVertexBuffers` recording the first binding, the count and the second entries.
unsafe extern "system" fn bind_vertex_buffers(
    _: vk::CommandBuffer,
    first: u32,
    count: u32,
    buffers: *const vk::Buffer,
    offsets: *const vk::DeviceSize,
) {
    see(u64::from(first));
    see(u64::from(count));
    // SAFETY: the wrapper passes `count` buffers and offsets.
    unsafe {
        see((*buffers.add(1)).as_raw());
        see(*offsets.add(1));
    }
}

/// `vkCmdBindVertexBuffers2` recording the count and whether sizes and strides are null.
unsafe extern "system" fn bind_vertex_buffers2(
    _: vk::CommandBuffer,
    _: u32,
    count: u32,
    _: *const vk::Buffer,
    _: *const vk::DeviceSize,
    sizes: *const vk::DeviceSize,
    strides: *const vk::DeviceSize,
) {
    see(u64::from(count));
    see(u64::from(sizes.is_null()));
    see(u64::from(strides.is_null()));
}

/// `vkCmdPushConstants` recording the size and the third `u32`.
unsafe extern "system" fn push_constants(
    _: vk::CommandBuffer,
    _: vk::PipelineLayout,
    _: vk::ShaderStageFlags,
    _: u32,
    size: u32,
    values: *const c_void,
) {
    see(u64::from(size));
    // SAFETY: the wrapper passes `size` bytes.
    see(u64::from(unsafe { *values.cast::<u32>().add(2) }));
}

/// A device with the three command-buffer mocks above.
fn command_device() -> Device {
    device(DeviceTable {
        cmd_bind_vertex_buffers: Some(bind_vertex_buffers),
        cmd_bind_vertex_buffers2: Some(bind_vertex_buffers2),
        cmd_push_constants: Some(push_constants),
        ..Default::default()
    })
}

#[test]
fn counts_are_slice_lengths_and_optional_arrays_null() {
    let d = command_device();
    let cb = vk::CommandBuffer::from_raw(9);
    let buffers = [vk::Buffer::from_raw(10), vk::Buffer::from_raw(11)];
    // SAFETY: the mocks read only the arrays the wrappers pass.
    unsafe {
        d.cmd_bind_vertex_buffers(cb, 3, &buffers, &[0, 64]);
        assert_eq!(seen(), [3, 2, 11, 64]);
        d.cmd_bind_vertex_buffers2(cb, 0, &buffers, &[0, 64], None, Some(&[4, 4]));
        assert_eq!(&seen()[4..], [2, 1, 0]);
        d.cmd_push_constants(
            cb,
            vk::PipelineLayout::null(),
            vk::ShaderStageFlags::VERTEX,
            0,
            &[5u32, 6, 7],
        );
        assert_eq!(
            &seen()[7..],
            [12, 7],
            "a byte array's count is its size in bytes"
        );
    }
}

#[test]
#[should_panic(expected = "bindingCount: arrays differ in length")]
fn arrays_sharing_a_count_must_agree() {
    let d = command_device();
    let buffers = [vk::Buffer::from_raw(10), vk::Buffer::from_raw(11)];
    // SAFETY: the wrapper panics before calling the mock.
    unsafe { d.cmd_bind_vertex_buffers(vk::CommandBuffer::from_raw(9), 0, &buffers, &[0]) };
}

/// `vkAllocateCommandBuffers` writing handle 5 into the second entry.
unsafe extern "system" fn allocate(
    _: vk::Device,
    _: *const vk::CommandBufferAllocateInfo<'_>,
    out: *mut vk::CommandBuffer,
) -> vk::Result {
    // SAFETY: the wrapper passes `commandBufferCount` handles.
    unsafe { *out.add(1) = vk::CommandBuffer::from_raw(5) };
    vk::Result::SUCCESS
}

/// `vkCmdSetSampleMaskEXT` recording whether the mask is null.
unsafe extern "system" fn sample_mask(
    _: vk::CommandBuffer,
    _: vk::SampleCountFlagBits,
    mask: *const vk::SampleMask,
) {
    see(u64::from(mask.is_null()));
}

#[test]
fn lengths_from_struct_members_and_expressions_are_asserted() {
    let d = device(DeviceTable {
        allocate_command_buffers: Some(allocate),
        cmd_set_sample_mask_ext: Some(sample_mask),
        ..Default::default()
    });
    let info = vk::CommandBufferAllocateInfo::default().command_buffer_count(2);
    let mut out = [vk::CommandBuffer::null(); 2];
    // SAFETY: the mocks touch only what the wrappers pass.
    unsafe {
        assert_eq!(d.allocate_command_buffers(&info, &mut out), Ok(()));
        assert_eq!(out[1], vk::CommandBuffer::from_raw(5));
        let cb = vk::CommandBuffer::from_raw(9);
        d.cmd_set_sample_mask_ext(cb, vk::SampleCountFlags::COUNT_64, Some(&[0, 0]));
        d.cmd_set_sample_mask_ext(cb, vk::SampleCountFlags::COUNT_4, None);
    }
    assert_eq!(seen(), [0, 1]);
}

#[test]
#[should_panic(expected = "slice length is not pAllocateInfo->commandBufferCount")]
fn a_member_length_mismatch_panics() {
    let d = device(DeviceTable {
        allocate_command_buffers: Some(allocate),
        ..Default::default()
    });
    let info = vk::CommandBufferAllocateInfo::default().command_buffer_count(3);
    let mut out = [vk::CommandBuffer::null(); 2];
    // SAFETY: the wrapper panics before calling the mock.
    let _ = unsafe { d.allocate_command_buffers(&info, &mut out) };
}

#[test]
#[should_panic(expected = "slice length is not (samples + 31) / 32")]
fn an_expression_length_mismatch_panics() {
    let d = device(DeviceTable {
        cmd_set_sample_mask_ext: Some(sample_mask),
        ..Default::default()
    });
    let cb = vk::CommandBuffer::from_raw(9);
    // SAFETY: the wrapper panics before calling the mock.
    unsafe { d.cmd_set_sample_mask_ext(cb, vk::SampleCountFlags::COUNT_4, Some(&[0, 0])) };
}

/// `vkMapMemory2` recording the size and returning address `0x4000`.
unsafe extern "system" fn map(
    _: vk::Device,
    info: *const vk::MemoryMapInfo<'_>,
    data: *mut *mut c_void,
) -> vk::Result {
    // SAFETY: the wrapper passes a live info and a live output.
    unsafe {
        see((*info).size);
        *data = ptr::without_provenance_mut(0x4000);
    }
    vk::Result::SUCCESS
}

#[test]
fn a_void_pointer_output_is_returned() {
    let d = device(DeviceTable {
        map_memory2: Some(map),
        ..Default::default()
    });
    let info = vk::MemoryMapInfo::default().size(512);
    // SAFETY: the mock reads the info and writes the output pointer.
    let mapped = unsafe { d.map_memory2(&info) };
    assert_eq!(mapped.map(|p| p.addr()), Ok(0x4000));
    assert_eq!(seen(), [512]);
}

/// `vkGetPhysicalDeviceSurfaceSupportKHR` writing `VK_TRUE`.
unsafe extern "system" fn surface_support(
    _: vk::PhysicalDevice,
    _: u32,
    _: vk::SurfaceKHR,
    supported: *mut vk::Bool32,
) -> vk::Result {
    // SAFETY: the wrapper passes a live output.
    unsafe { *supported = vk::TRUE };
    vk::Result::SUCCESS
}

#[test]
fn a_returned_bool32_is_a_bool() {
    let instance = Instance::from_parts(
        vk::Instance::from_raw(1),
        InstanceTable {
            get_physical_device_surface_support_khr: Some(surface_support),
            ..Default::default()
        },
    );
    // SAFETY: the mock writes only the output the wrapper passes.
    let supported = unsafe {
        instance.get_physical_device_surface_support_khr(
            vk::PhysicalDevice::from_raw(2),
            0,
            vk::SurfaceKHR::from_raw(3),
        )
    };
    assert_eq!(supported, Ok(true));
}

/// `vkGetDeviceProcAddr` that knows only the KHR name of `vkCmdDrawIndirectCount`.
unsafe extern "system" fn device_proc_addr(
    _: vk::Device,
    name: *const c_char,
) -> Option<vk::PfnVoidFunction> {
    // SAFETY: the table loader passes NUL-terminated names.
    let name = unsafe { CStr::from_ptr(name) };
    (name == c"vkCmdDrawIndirectCountKHR").then_some(void as vk::PfnVoidFunction)
}

/// A command that does nothing, for the table to hold.
unsafe extern "system" fn void() {}

#[test]
fn device_commands_load_through_the_instance_with_alias_fallback() {
    let instance = Instance::from_parts(
        vk::Instance::from_raw(1),
        InstanceTable {
            get_device_proc_addr: Some(device_proc_addr),
            ..Default::default()
        },
    );
    // SAFETY: the mock returns a function only for the alias name.
    let d = unsafe { Device::load(&instance, vk::Device::from_raw(2)) };
    assert!(
        d.table().cmd_draw_indirect_count.is_some(),
        "loaded by its KHR alias"
    );
    assert!(d.table().cmd_draw.is_none());
    assert_eq!(d.handle(), vk::Device::from_raw(2));
}

/// `vkCreateInstance` counting the call and creating instance `0x10`.
unsafe extern "system" fn create_instance(
    _: *const vk::InstanceCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Instance,
) -> vk::Result {
    call();
    // SAFETY: the runtime passes a live output.
    unsafe { *out = vk::Instance::from_raw(0x10) };
    vk::Result::SUCCESS
}

/// `vkCreateDevice` counting the call and creating device `0x20`.
unsafe extern "system" fn create_device(
    _: vk::PhysicalDevice,
    _: *const vk::DeviceCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Device,
) -> vk::Result {
    call();
    // SAFETY: the runtime passes a live output.
    unsafe { *out = vk::Device::from_raw(0x20) };
    vk::Result::SUCCESS
}

#[test]
fn a_missing_loader_panics_before_anything_is_created() {
    let entry = Entry::from_parts(EntryTable {
        create_instance: Some(create_instance),
        ..Default::default()
    });
    // SAFETY: the runtime panics before calling the mock, which writes only its output.
    let created = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default(), None)
    }));
    assert!(created.is_err(), "no vkGetInstanceProcAddr: a panic");
    let instance = Instance::from_parts(
        vk::Instance::from_raw(1),
        InstanceTable {
            create_device: Some(create_device),
            ..Default::default()
        },
    );
    // SAFETY: as above.
    let created = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        instance.create_device(
            vk::PhysicalDevice::from_raw(2),
            &vk::DeviceCreateInfo::default(),
            None,
        )
    }));
    assert!(created.is_err(), "no vkGetDeviceProcAddr: a panic");
    assert_eq!(calls(), 0, "neither object was created");
}

#[test]
#[should_panic(expected = "vkCmdDraw is not loaded")]
fn an_unloaded_command_panics_with_its_name() {
    let d = device(DeviceTable::default());
    // SAFETY: the wrapper panics before any call.
    unsafe { d.cmd_draw(vk::CommandBuffer::from_raw(1), 3, 1, 0, 0) };
}

/// A consuming command (`vkGetPastPresentationTimingGOOGLE`): each call returns the oldest
/// records that fit and forgets them; `INCOMPLETE` while more remain.
unsafe extern "system" fn past_timing(
    _: vk::Device,
    _: vk::SwapchainKHR,
    count: *mut u32,
    timings: *mut vk::PastPresentationTimingGOOGLE,
) -> vk::Result {
    call();
    RECORDS.with(|r| {
        let mut records = r.borrow_mut();
        // SAFETY: the wrapper passes a live count and an array of `*count` timings.
        unsafe {
            let n = records.len().min(*count as usize);
            for (i, id) in records.drain(..n).enumerate() {
                (*timings.add(i)).present_id = id;
            }
            *count = n as u32;
        }
        if records.is_empty() {
            vk::Result::SUCCESS
        } else {
            vk::Result::INCOMPLETE
        }
    })
}

#[test]
fn a_consuming_command_fills_caller_storage_once_per_call() {
    RECORDS.with(|r| *r.borrow_mut() = vec![1, 2]);
    let d = device(DeviceTable {
        get_past_presentation_timing_google: Some(past_timing),
        ..Default::default()
    });
    let swapchain = vk::SwapchainKHR::from_raw(1);
    let mut timings = [vk::PastPresentationTimingGOOGLE::default()];
    // SAFETY: the mock writes at most the slice's length of records.
    let first = unsafe { d.get_past_presentation_timing_google_into(swapchain, &mut timings) };
    assert_eq!(first, Ok((1, vk::Result::INCOMPLETE)));
    assert_eq!(timings[0].present_id, 1);
    // SAFETY: as above.
    let second = unsafe { d.get_past_presentation_timing_google_into(swapchain, &mut timings) };
    assert_eq!(second, Ok((1, vk::Result::SUCCESS)));
    assert_eq!(timings[0].present_id, 2, "record 1 is not returned again");
    assert_eq!(calls(), 2, "one call each, no count query, no retry");
}

#[test]
#[should_panic(
    expected = "pPresentationTimings: empty slice; it needs room for at least one record"
)]
fn empty_caller_storage_panics() {
    let d = device(DeviceTable {
        get_past_presentation_timing_google: Some(past_timing),
        ..Default::default()
    });
    // SAFETY: the wrapper panics before calling the mock.
    let _ =
        unsafe { d.get_past_presentation_timing_google_into(vk::SwapchainKHR::null(), &mut []) };
}

/// A blocking consuming command (`vkGetDeviceFaultReportsKHR`) that times out with no reports.
unsafe extern "system" fn fault_reports(
    _: vk::Device,
    timeout: u64,
    count: *mut u32,
    _: *mut vk::DeviceFaultInfoKHR<'_>,
) -> vk::Result {
    call();
    see(timeout);
    // SAFETY: the wrapper passes a live count.
    unsafe { *count = 0 };
    vk::Result::TIMEOUT
}

#[test]
fn a_blocking_command_waits_once() {
    let d = device(DeviceTable {
        get_device_fault_reports_khr: Some(fault_reports),
        ..Default::default()
    });
    let mut reports = [vk::DeviceFaultInfoKHR::default(); 2];
    // SAFETY: the mock writes no report.
    let got = unsafe { d.get_device_fault_reports_khr_into(7, &mut reports) };
    assert_eq!(got, Ok((0, vk::Result::TIMEOUT)));
    assert_eq!(seen(), [7], "the timeout is passed once");
    assert_eq!(calls(), 1, "a timeout is returned, never retried");
}

/// `vkGetPhysicalDeviceQueueFamilyProperties2` with two families: the count, or as many families
/// as fit, each writing a priority count through the caller's `pNext`.
unsafe extern "system" fn family_properties(
    _: vk::PhysicalDevice,
    count: *mut u32,
    properties: *mut vk::QueueFamilyProperties2<'_>,
) {
    call();
    // SAFETY: the wrapper passes a live count and, when non-null, `*count` properties whose
    // `pNext` the caller set.
    unsafe {
        if properties.is_null() {
            *count = 2;
            return;
        }
        let n = (*count).min(2);
        for i in 0..n as usize {
            let p = &mut *properties.add(i);
            p.queue_family_properties.queue_count = 4;
            let next = p
                .p_next
                .cast::<vk::QueueFamilyGlobalPriorityProperties<'_>>();
            if !next.is_null() {
                (*next).priority_count = 3;
            }
        }
        *count = n;
    }
}

#[test]
fn an_extensible_query_keeps_the_callers_chains() {
    let instance = Instance::from_parts(
        vk::Instance::from_raw(1),
        InstanceTable {
            get_physical_device_queue_family_properties2: Some(family_properties),
            ..Default::default()
        },
    );
    let gpu = vk::PhysicalDevice::from_raw(2);
    // SAFETY: the mock writes only the count.
    let len = unsafe { instance.get_physical_device_queue_family_properties2_len(gpu) };
    assert_eq!(len, 2);
    let mut priority = vk::QueueFamilyGlobalPriorityProperties::default();
    let written = {
        let mut properties = [vk::QueueFamilyProperties2::default().push_next(&mut priority)];
        // SAFETY: the mock writes one family and its chained struct.
        let written = unsafe {
            instance.get_physical_device_queue_family_properties2_into(gpu, &mut properties)
        };
        assert_eq!(properties[0].queue_family_properties.queue_count, 4);
        written
    };
    assert_eq!(written, 1, "as many as the storage holds");
    assert_eq!(
        priority.priority_count, 3,
        "written through the caller's pNext"
    );
    assert_eq!(calls(), 2, "the count query, then one fill");
}

/// `vkGetQueryPoolResults` writing nothing.
unsafe extern "system" fn query_results(
    _: vk::Device,
    _: vk::QueryPool,
    _: u32,
    _: u32,
    _: usize,
    _: *mut c_void,
    _: vk::DeviceSize,
    _: vk::QueryResultFlags,
) -> vk::Result {
    vk::Result::SUCCESS
}

/// Four `u32`s aligned to 8: the second is aligned to 4 only.
#[repr(C, align(8))]
struct Results([u32; 4]);

#[test]
fn query_results_are_aligned_as_their_flags_require() {
    let d = device(DeviceTable {
        get_query_pool_results: Some(query_results),
        ..Default::default()
    });
    let mut results = Results([0; 4]);
    let pool = vk::QueryPool::from_raw(1);
    // SAFETY: the mock reads nothing.
    unsafe {
        let four = d.get_query_pool_results(
            pool,
            0,
            1,
            &mut results.0[1..],
            4,
            vk::QueryResultFlags::empty(),
        );
        assert_eq!(four, Ok(vk::Result::SUCCESS), "4-aligned 32-bit results");
        let eight = d.get_query_pool_results(
            pool,
            0,
            1,
            &mut results.0[..2],
            8,
            vk::QueryResultFlags::RESULT_64,
        );
        assert_eq!(eight, Ok(vk::Result::SUCCESS), "8-aligned 64-bit results");
    }
}

#[test]
#[should_panic(expected = "pData: address not aligned to 8 bytes with VK_QUERY_RESULT_64_BIT")]
fn misaligned_64_bit_query_results_panic() {
    let d = device(DeviceTable {
        get_query_pool_results: Some(query_results),
        ..Default::default()
    });
    let mut results = Results([0; 4]);
    // SAFETY: the wrapper panics before calling the mock.
    let _ = unsafe {
        d.get_query_pool_results(
            vk::QueryPool::from_raw(1),
            0,
            1,
            &mut results.0[1..3],
            8,
            vk::QueryResultFlags::RESULT_64,
        )
    };
}

/// `vkCmdUpdateBuffer` recording the size.
unsafe extern "system" fn update_buffer(
    _: vk::CommandBuffer,
    _: vk::Buffer,
    _: vk::DeviceSize,
    size: vk::DeviceSize,
    _: *const c_void,
) {
    see(size);
}

#[test]
fn generated_plain_structs_are_byte_data() {
    let d = device(DeviceTable {
        cmd_update_buffer: Some(update_buffer),
        ..Default::default()
    });
    let draws = [vk::DrawIndexedIndirectCommand::default().index_count(3); 2];
    // SAFETY: the mock reads only the size.
    unsafe {
        d.cmd_update_buffer(
            vk::CommandBuffer::from_raw(1),
            vk::Buffer::from_raw(2),
            0,
            &draws,
        );
    }
    assert_eq!(seen(), [40], "two 20-byte commands");
}
