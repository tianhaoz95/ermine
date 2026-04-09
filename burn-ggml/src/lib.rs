pub mod backend;
pub mod context;
pub mod device;
pub mod gguf;
pub mod graph;
pub mod memory;
pub mod model;
pub mod ops;
pub mod tensor;

pub use backend::*;
pub use context::*;
pub use device::*;
pub use ops::module_ops::GgmlOps;
pub use ops::offload::*;
pub use ops::prefetch::*;
pub use ops::tokenizer::*;
pub use tensor::*;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance};

    #[tokio::test]
    async fn test_add() {
        let device = GgmlDevice::Cpu;
        let lhs = Tensor::<GgmlBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let rhs = Tensor::<GgmlBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);

        let sum = lhs + rhs;
        let data = sum.into_data();

        data.assert_approx_eq(
            &TensorData::from([[6.0, 8.0], [10.0, 12.0]]),
            Tolerance::<f32>::absolute(1e-5),
        );
    }

    #[tokio::test]
    async fn test_mul() {
        let device = GgmlDevice::Cpu;
        let lhs = Tensor::<GgmlBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let rhs = Tensor::<GgmlBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);

        let res = lhs * rhs;
        let data = res.into_data();

        data.assert_approx_eq(
            &TensorData::from([[5.0, 12.0], [21.0, 32.0]]),
            Tolerance::<f32>::absolute(1e-5),
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_matmul() {
        let device = GgmlDevice::Cpu;
        let lhs = Tensor::<GgmlBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let rhs = Tensor::<GgmlBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);

        let res = lhs.matmul(rhs);
        let data = res.into_data();

        // [1 2] * [5 6] = [1*5+2*7 1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7 3*6+4*8]   [43 50]
        data.assert_approx_eq(
            &TensorData::from([[19.0, 22.0], [43.0, 50.0]]),
            Tolerance::<f32>::absolute(1e-5),
        );
    }
}
