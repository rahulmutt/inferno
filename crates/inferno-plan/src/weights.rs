use crate::{PlanError, Result};
use inferno_formats::{DType, ModelDesc, TensorDesc, quant, read_tensor_bytes};
use inferno_graph::{Graph, Op};
use inferno_kernels::{AlignedBuf, KernelIsa, kernels_for, reference_kernels};
use inferno_target::TargetDesc;

/// A single MatMul weight matrix, repacked into the "rs8" kernel layout and
/// placed at a byte offset within the shared weight image (spec
/// §rs8-packed weight image).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedWeight {
    /// Index into `desc.tensors`.
    pub tensor_index: usize,
    /// Byte offset into the weight image.
    pub offset: usize,
    /// Packed byte length.
    pub len: usize,
    pub rows: usize,
    pub k: usize,
    pub dtype: DType,
    pub isa: KernelIsa,
    /// Packed-layout identifier; always "rs8" in M3.
    pub layout: &'static str,
}

/// The concatenated packed-weight blob (destined for `weights.bin`) plus the
/// per-weight index into it, in graph MatMul order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WeightImageLayout {
    pub image: Vec<u8>,
    pub weights: Vec<PackedWeight>,
}

/// Pack one weight tensor into the rs8 layout, returning the packed buffer
/// alongside the *stored* dtype/isa/layout metadata for its `PackedWeight`.
///
/// Native-kernel dtypes (F32/Q8_0/Q4_K) pack directly. F16/BF16 have no
/// kernel, so they are widened at compile time to f32 — dequantized with the
/// *same* [`quant::dequant`] helper the interpreter uses (interp.rs:84), so
/// the compiled f32 matmul matches the scalar oracle bit-for-bit — then
/// packed through the f32 kernel. The returned `dtype` is then `F32` (not the
/// on-disk F16/BF16): Task 7 keys the gemv symbol and the activation-quantize
/// skip on this *stored* dtype, so it must reflect what actually got packed.
fn pack_weight(
    desc: &ModelDesc,
    td: &TensorDesc,
    target: &TargetDesc,
    rows: usize,
    k: usize,
) -> Result<(AlignedBuf, DType, KernelIsa, &'static str)> {
    let bytes = read_tensor_bytes(desc, td)?;
    if let Some(ks) = kernels_for(&td.dtype, target.isa).or_else(|| reference_kernels(&td.dtype)) {
        let packed = ks.pack(&bytes, rows, k)?;
        return Ok((packed, td.dtype.clone(), ks.isa, ks.layout));
    }
    // No native kernel. Widen F16/BF16 to f32 and pack with the f32 kernel;
    // any other dtype genuinely has no path and is an error.
    match td.dtype {
        DType::F16 | DType::BF16 => {
            let vals = quant::dequant(&td.dtype, &bytes, rows * k)?;
            let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ks = kernels_for(&DType::F32, target.isa)
                .or_else(|| reference_kernels(&DType::F32))
                .ok_or(PlanError::NoKernel(DType::F32))?;
            let packed = ks.pack(&f32_bytes, rows, k)?;
            Ok((packed, DType::F32, ks.isa, ks.layout))
        }
        _ => Err(PlanError::NoKernel(td.dtype.clone())),
    }
}

/// Walk the graph's `MatMul` nodes in order, repacking each weight matrix
/// through the target-appropriate kernel (falling back to the scalar
/// reference kernel when no SIMD kernel is available, and to compile-time f32
/// widening for F16/BF16 weights, which have no kernel) and appending it to a
/// single contiguous image. Bias vectors are *not* packed here — they stay
/// plain f32 tensors read at codegen time (Task 10).
pub fn build_weight_image(
    desc: &ModelDesc,
    graph: &Graph,
    target: &TargetDesc,
) -> Result<WeightImageLayout> {
    let mut layout = WeightImageLayout::default();
    for node in &graph.nodes {
        let Op::MatMul { weight, .. } = &node.op else {
            continue;
        };
        let td = &desc.tensors[weight.0];
        if td.shape.len() != 2 {
            return Err(PlanError::BadWeightRank {
                name: td.name.clone(),
                rank: td.shape.len(),
            });
        }
        let rows = td.shape[0] as usize;
        let k = td.shape[1] as usize;
        let (packed, dtype, isa, pack_layout) = pack_weight(desc, td, target, rows, k)?;
        let offset = layout.image.len();
        let len = packed.len();
        layout.image.extend_from_slice(packed.as_slice());
        layout.weights.push(PackedWeight {
            tensor_index: weight.0,
            offset,
            len,
            rows,
            k,
            dtype,
            isa,
            layout: pack_layout,
        });
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    fn setup() -> (inferno_formats::ModelDesc, inferno_graph::Graph, TargetDesc) {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        (desc, graph, target)
    }

    #[test]
    fn one_packed_weight_per_matmul() {
        let (desc, graph, target) = setup();
        let matmuls = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, inferno_graph::Op::MatMul { .. }))
            .count();
        let img = build_weight_image(&desc, &graph, &target).unwrap();
        assert_eq!(img.weights.len(), matmuls);
    }

    #[test]
    fn offsets_are_contiguous_and_match_image_len() {
        let (desc, graph, target) = setup();
        let img = build_weight_image(&desc, &graph, &target).unwrap();
        let mut cursor = 0usize;
        for w in &img.weights {
            assert_eq!(w.offset, cursor, "weight {} not contiguous", w.tensor_index);
            assert!(
                w.len > 0,
                "weight {} has zero packed length",
                w.tensor_index
            );
            cursor += w.len;
        }
        assert_eq!(cursor, img.image.len());
    }

    #[test]
    fn packed_bytes_equal_kernel_pack() {
        // The image slice for each weight must be byte-identical to packing
        // that weight's input bytes through the kernel selected for its
        // *stored* dtype — no reordering surprises. Reconstructed independently
        // of the image (from the tensor bytes, re-deriving the widening for
        // F16/BF16) so it still catches offset/length/reordering bugs; it is
        // NOT a build_weight_image-called-twice tautology.
        let (desc, graph, target) = setup();
        let img = build_weight_image(&desc, &graph, &target).unwrap();

        // The fixture must exercise both paths for this test to be meaningful.
        let mut saw_native = false;
        let mut saw_widened = false;

        for w in &img.weights {
            let td = &desc.tensors[w.tensor_index];
            let raw = inferno_formats::read_tensor_bytes(&desc, td).unwrap();

            // Rebuild the exact input bytes the packer saw, keyed on the
            // relationship between the stored dtype and the on-disk dtype.
            let input: Vec<u8> = if w.dtype == td.dtype {
                saw_native = true;
                raw
            } else {
                // Widened: stored F32, on-disk F16/BF16.
                assert_eq!(w.dtype, inferno_formats::DType::F32);
                assert!(matches!(
                    td.dtype,
                    inferno_formats::DType::F16 | inferno_formats::DType::BF16
                ));
                saw_widened = true;
                let vals = inferno_formats::quant::dequant(&td.dtype, &raw, w.rows * w.k).unwrap();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            };

            let ks = inferno_kernels::kernels_for(&w.dtype, target.isa)
                .or_else(|| inferno_kernels::reference_kernels(&w.dtype))
                .unwrap();
            let direct = ks.pack(&input, w.rows, w.k).unwrap();
            assert_eq!(&img.image[w.offset..w.offset + w.len], direct.as_slice());
        }

        assert!(saw_native, "fixture must have a natively-packed weight");
        assert!(saw_widened, "fixture must have a widened F16/BF16 weight");
    }
}
