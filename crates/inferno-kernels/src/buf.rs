//! 32-byte-aligned byte buffers: packed weight images satisfy the kernels'
//! aligned-load contract by construction, so callers can't get it wrong.

/// One aligned lane; `Vec<Lane>` keeps the whole allocation 32-byte aligned.
#[derive(Clone)]
#[repr(C, align(32))]
struct Lane([u8; 32]);

#[derive(Clone)]
pub struct AlignedBuf {
    lanes: Vec<Lane>,
    len: usize,
}

impl AlignedBuf {
    pub fn zeroed(len: usize) -> Self {
        AlignedBuf {
            lanes: vec![Lane([0u8; 32]); len.div_ceil(32)],
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.lanes.as_ptr().cast()
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `lanes` is one contiguous allocation of lanes.len()*32 >= len
        // initialized bytes, and the cast pointer lives as long as &self.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as as_slice, and &mut self guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.lanes.as_mut_ptr().cast(), self.len) }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_and_sized() {
        for len in [0usize, 1, 31, 32, 33, 4096] {
            let mut b = AlignedBuf::zeroed(len);
            assert_eq!(b.as_ptr() as usize % 32, 0);
            assert_eq!(b.as_slice().len(), len);
            assert_eq!(b.as_mut_slice().len(), len);
            assert!(b.as_slice().iter().all(|&x| x == 0));
        }
    }
}
