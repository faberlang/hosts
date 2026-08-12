//! Crate-local C ABI context carrier for the LLVM host runtime.
//!
//! [`FaberRtContextV1`] is the opaque process-lifetime runtime context: only
//! pointers cross the ABI, so the struct is a zero-sized opaque carrier.
//!
//! Authority note (faber-target-runtime inventory §3.4): the stable C ABI is
//! compiler-owned — symbol names, status codes, value kinds, and the ABI
//! structs' layouts are the `radix-host-abi` authority, which this crate
//! imports directly (`radix_host_abi`). This context carrier is the **LLVM
//! host side** of the ABI (the host runtime allocates the context and passes
//! it to emitted programs); it is carried here pending the radix-host-abi
//! contract migration, which should adopt the same opaque definition so the
//! two sides can never drift. The historical definition lived in
//! `faber-runtime/src/host_abi.rs` (migration carrier).

use core::ffi::c_void;

/// Opaque process-lifetime runtime context. Only pointers cross the ABI.
#[repr(C)]
pub struct FaberRtContextV1 {
    _private: [u8; 0],
    _alignment: [*mut c_void; 0],
}
