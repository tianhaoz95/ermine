use burn::tensor::ops::ModuleOps;
use crate::{GgmlBackend, GgmlTensor, GgmlContext};
use burn::tensor::backend::ExecutionError;
use ggml_sys::*;
use std::sync::Arc;
use burn::tensor::ops::*;

impl ModuleOps<GgmlBackend> for GgmlBackend {
    fn embedding(_weight: GgmlTensor, _indices: GgmlTensor) -> GgmlTensor { todo!() }
    fn embedding_backward(_weight: GgmlTensor, _output: GgmlTensor, _indices: GgmlTensor) -> GgmlTensor { todo!() }
    fn conv1d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<1>) -> GgmlTensor { todo!() }
    fn conv2d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<2>) -> GgmlTensor { todo!() }
    fn deform_conv2d(_x: GgmlTensor, _offset: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _mask: Option<GgmlTensor>, _options: DeformConvOptions<2>) -> GgmlTensor { todo!() }
    fn deform_conv2d_backward(_x: GgmlTensor, _offset: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _mask: Option<GgmlTensor>, _out_grad: GgmlTensor, _options: DeformConvOptions<2>) -> DeformConv2dBackward<GgmlBackend> { todo!() }
    fn conv3d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<3>) -> GgmlTensor { todo!() }
    fn conv_transpose2d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvTransposeOptions<2>) -> GgmlTensor { todo!() }
    fn conv_transpose3d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvTransposeOptions<3>) -> GgmlTensor { todo!() }
    fn avg_pool2d(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _count_include_pad: bool, _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn avg_pool2d_backward(_x: GgmlTensor, _grad: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _count_include_pad: bool, _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn adaptive_avg_pool2d(_x: GgmlTensor, _output_size: [usize; 2]) -> GgmlTensor { todo!() }
    fn adaptive_avg_pool2d_backward(_x: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor { todo!() }
    fn max_pool2d(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn max_pool2d_with_indices(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool) -> MaxPool2dWithIndices<GgmlBackend> { todo!() }
    fn max_pool2d_with_indices_backward(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool, _output_grad: GgmlTensor, _indices: GgmlTensor) -> MaxPool2dBackward<GgmlBackend> { todo!() }
    fn interpolate(_x: GgmlTensor, _output_size: [usize; 2], _options: InterpolateOptions) -> GgmlTensor { todo!() }
    fn interpolate_backward(_x: GgmlTensor, _grad: GgmlTensor, _output_size: [usize; 2], _options: InterpolateOptions) -> GgmlTensor { todo!() }
    fn attention(_q: GgmlTensor, _k: GgmlTensor, _v: GgmlTensor, _mask: Option<GgmlTensor>, _dropout: Option<GgmlTensor>, _options: AttentionModuleOptions) -> GgmlTensor { todo!() }
}
