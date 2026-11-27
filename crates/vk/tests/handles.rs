//! Dispatchable handles are pointers end to end: a mock `vkCreateDevice` returns a pointer to a
//! real allocation, and the bindings carry it, and the queue and command buffer pointers the mocks
//! derive from it, through `Instance::create_device`, the device table loader and the wrappers.
//! Every later mock dereferences the pointer it receives and records the tag it reads there. Under
//! Miri each read is checked against the allocation's provenance, and the box `vkDestroyDevice`
//! frees must leave nothing leaked. Mocks record into thread-local state and never assert: a
//! panic cannot unwind out of an `extern "system"` function.

use core::cell::{Cell, RefCell};
use core::ffi::{CStr, c_char, c_void};
use core::{mem, ptr};
use tessera_vk::{Instance, InstanceTable, vk};

/// What every mock dispatchable object starts with: a tag the mocks read back.
#[repr(C)]
struct MockObject {
    /// Which object this is.
    tag: u64,
}

/// The mock device: its own tagged object first (so a device pointer is also a `MockObject`
/// pointer), then the queue and command buffers the mocks hand out as pointers into it.
#[repr(C)]
struct MockDevice {
    /// The device's own object.
    object: MockObject,
    /// The one queue.
    queue: MockObject,
    /// The command buffers `vkAllocateCommandBuffers` hands out.
    command_buffers: [MockObject; 2],
}

/// The device's tag.
const DEVICE: u64 = 0xD0;
/// The queue's tag.
const QUEUE: u64 = 0x51;
/// The command buffers' tags.
const COMMAND_BUFFERS: [u64; 2] = [0xC0, 0xC1];

thread_local! {
    /// `(call, tag read through the handle)` for every call after the device's creation.
    static SEEN: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) };
    /// `(lookups, lookups whose device pointer led to the device's tag)` of the proc-addr mock.
    static LOOKUPS: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

/// Records that `call` read `tag`.
fn see(call: &'static str, tag: u64) {
    SEEN.with(|s| s.borrow_mut().push((call, tag)));
}

/// The tag of the mock object at dispatchable handle pointer `raw`.
///
/// # Safety
///
/// `raw` points to a live `MockObject`, or a live `MockDevice`, which starts with one.
unsafe fn tag(raw: *mut c_void) -> u64 {
    // SAFETY: the caller guarantees a live object at `raw`; the read goes through the raw
    // pointer, with no reference that could invalidate other pointers into the allocation.
    unsafe { (*raw.cast::<MockObject>()).tag }
}

/// `vkCreateDevice`: the device is a boxed `MockDevice`, its handle the box's pointer.
unsafe extern "system" fn create_device(
    _: vk::PhysicalDevice,
    _: *const vk::DeviceCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Device,
) -> vk::Result {
    let device = Box::into_raw(Box::new(MockDevice {
        object: MockObject { tag: DEVICE },
        queue: MockObject { tag: QUEUE },
        command_buffers: COMMAND_BUFFERS.map(|tag| MockObject { tag }),
    }));
    // SAFETY: the runtime passes a live output.
    unsafe { *out = vk::Device::from_raw(device.cast()) };
    vk::Result::SUCCESS
}

/// `vkGetDeviceProcAddr`: reads the device's tag on every lookup and knows the commands below.
unsafe extern "system" fn device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> Option<vk::PfnVoidFunction> {
    // SAFETY: the loader passes the live device it was created with.
    let device_tag = unsafe { tag(device.as_raw()) };
    LOOKUPS.with(|l| {
        let (lookups, matched) = l.get();
        l.set((lookups + 1, matched + u32::from(device_tag == DEVICE)));
    });
    // SAFETY: the table loader passes NUL-terminated names.
    let name = unsafe { CStr::from_ptr(name) };
    // SAFETY: each transmute turns a function pointer into a function pointer of the same size;
    // the table loader casts it back to the type its name stands for, which is the mock's.
    unsafe {
        Some(match name.to_bytes() {
            b"vkGetDeviceQueue" => {
                mem::transmute::<vk::PfnGetDeviceQueue, vk::PfnVoidFunction>(get_device_queue)
            }
            b"vkAllocateCommandBuffers" => mem::transmute::<
                vk::PfnAllocateCommandBuffers,
                vk::PfnVoidFunction,
            >(allocate_command_buffers),
            b"vkQueueSubmit2" => {
                mem::transmute::<vk::PfnQueueSubmit2, vk::PfnVoidFunction>(queue_submit2)
            }
            b"vkFreeCommandBuffers" => mem::transmute::<
                vk::PfnFreeCommandBuffers,
                vk::PfnVoidFunction,
            >(free_command_buffers),
            b"vkDestroyDevice" => {
                mem::transmute::<vk::PfnDestroyDevice, vk::PfnVoidFunction>(destroy_device)
            }
            _ => return None,
        })
    }
}

/// `vkGetDeviceQueue`: the queue is a pointer to the device's `queue` field.
unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    _: u32,
    _: u32,
    out: *mut vk::Queue,
) {
    // SAFETY: the wrapper passes the live device and a live output; the field pointer is derived
    // from the device's own pointer, so it carries the box's provenance.
    unsafe {
        see("queue", tag(device.as_raw()));
        let device = device.as_raw().cast::<MockDevice>();
        *out = vk::Queue::from_raw((&raw mut (*device).queue).cast());
    }
}

/// `vkAllocateCommandBuffers`: each command buffer is a pointer into the device's
/// `command_buffers`.
unsafe extern "system" fn allocate_command_buffers(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    out: *mut vk::CommandBuffer,
) -> vk::Result {
    // SAFETY: the wrapper passes the live device, a live allocate info and `commandBufferCount`
    // outputs; at most two are written, each a pointer derived from the device's own.
    unsafe {
        see("allocate", tag(device.as_raw()));
        let device = device.as_raw().cast::<MockDevice>();
        let first = (&raw mut (*device).command_buffers).cast::<MockObject>();
        let count = ((*info).command_buffer_count as usize).min(COMMAND_BUFFERS.len());
        for i in 0..count {
            *out.add(i) = vk::CommandBuffer::from_raw(first.add(i).cast());
        }
    }
    vk::Result::SUCCESS
}

/// `vkQueueSubmit2`: reads the queue's tag, then each submitted command buffer's.
unsafe extern "system" fn queue_submit2(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo2<'_>,
    _: vk::Fence,
) -> vk::Result {
    // SAFETY: the wrapper passes the live queue and `count` live submits, each with
    // `commandBufferInfoCount` live command buffer infos naming live command buffers.
    unsafe {
        see("submit queue", tag(queue.as_raw()));
        for s in 0..count as usize {
            let submit = &*submits.add(s);
            for i in 0..submit.command_buffer_info_count as usize {
                let buffer = (*submit.p_command_buffer_infos.add(i)).command_buffer;
                see("submit command buffer", tag(buffer.as_raw()));
            }
        }
    }
    vk::Result::SUCCESS
}

/// `vkFreeCommandBuffers`: reads each command buffer's tag.
unsafe extern "system" fn free_command_buffers(
    _: vk::Device,
    _: vk::CommandPool,
    count: u32,
    buffers: *const vk::CommandBuffer,
) {
    // SAFETY: the wrapper passes `count` live command buffers.
    unsafe {
        for i in 0..count as usize {
            see("free", tag((*buffers.add(i)).as_raw()));
        }
    }
}

/// `vkDestroyDevice`: reads the device's tag and frees its box.
unsafe extern "system" fn destroy_device(
    device: vk::Device,
    _: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: the wrapper passes the live device, which `create_device` boxed; nothing uses it
    // or a pointer into it afterwards.
    unsafe {
        see("destroy", tag(device.as_raw()));
        drop(Box::from_raw(device.as_raw().cast::<MockDevice>()));
    }
}

#[test]
fn dispatchable_handles_carry_their_pointers_through_the_bindings() {
    let instance = Instance::from_parts(
        // A synthetic instance with no provenance: no mock dereferences it.
        vk::Instance::from_raw(ptr::without_provenance_mut(1)),
        InstanceTable {
            create_device: Some(create_device),
            get_device_proc_addr: Some(device_proc_addr),
            ..Default::default()
        },
    );
    // SAFETY: the mocks dereference only the pointers they create, while the device lives.
    let device = unsafe {
        instance.create_device(
            vk::PhysicalDevice::from_raw(ptr::without_provenance_mut(2)),
            &vk::DeviceCreateInfo::default(),
            None,
        )
    }
    .expect("the mock creates the device");
    let (lookups, matched) = LOOKUPS.with(Cell::get);
    assert!(
        lookups > 0,
        "the table is loaded through vkGetDeviceProcAddr"
    );
    assert_eq!(matched, lookups, "every lookup reached the device");
    let mut buffers = [vk::CommandBuffer::null(); 2];
    // SAFETY: as above; each call passes handles the mocks created.
    unsafe {
        let queue = device.get_device_queue(0, 0);
        let info = vk::CommandBufferAllocateInfo::default().command_buffer_count(2);
        device
            .allocate_command_buffers(&info, &mut buffers)
            .expect("the mock allocates");
        let infos = buffers.map(|b| vk::CommandBufferSubmitInfo::default().command_buffer(b));
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&infos)];
        device
            .queue_submit2(queue, &submits, vk::Fence::null())
            .expect("the mock submits");
        device.free_command_buffers(vk::CommandPool::null(), &buffers);
        device.destroy_device(None);
    }
    assert!(buffers.iter().all(|b| !b.is_null()) && buffers[0] != buffers[1]);
    let seen = SEEN.with(|s| s.borrow().clone());
    assert_eq!(
        seen,
        [
            ("queue", DEVICE),
            ("allocate", DEVICE),
            ("submit queue", QUEUE),
            ("submit command buffer", COMMAND_BUFFERS[0]),
            ("submit command buffer", COMMAND_BUFFERS[1]),
            ("free", COMMAND_BUFFERS[0]),
            ("free", COMMAND_BUFFERS[1]),
            ("destroy", DEVICE),
        ],
        "every mock read its object's tag through the handle it received"
    );
}
