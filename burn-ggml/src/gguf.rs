use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;
use byteorder::{ReadBytesExt, LittleEndian};
use std::io::{Cursor, Seek, SeekFrom, Read};
use std::sync::Arc;
use crate::context::GgmlContext;
use crate::tensor::GgmlTensor;
use ggml_sys::*;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic,
    InvalidVersion(u32),
    TensorNotFound(String),
}

pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: ggml_type,
    pub data_offset: u64,
    pub data_size: usize,
}

pub struct GgufIndex {
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data_section_offset: u64,
    pub mmap: Mmap,
}

#[derive(Debug)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
}

impl GgufMetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufMetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }
}

impl GgufIndex {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path).map_err(GgufError::Io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(GgufError::Io)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>().map_err(GgufError::Io)?;
        if magic != 0x46554747 {
            return Err(GgufError::InvalidMagic);
        }

        let version = cursor.read_u32::<LittleEndian>().map_err(GgufError::Io)?;
        if version != 2 && version != 3 {
            return Err(GgufError::InvalidVersion(version));
        }

        let n_tensors = cursor.read_u64::<LittleEndian>().map_err(GgufError::Io)?;
        let n_kv = cursor.read_u64::<LittleEndian>().map_err(GgufError::Io)?;

        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut cursor).map_err(GgufError::Io)?;
            let val_type = cursor.read_u32::<LittleEndian>().map_err(GgufError::Io)?;
            let val = read_value(&mut cursor, val_type).map_err(GgufError::Io)?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = read_string(&mut cursor).map_err(GgufError::Io)?;
            let n_dims = cursor.read_u32::<LittleEndian>().map_err(GgufError::Io)?;
            let mut shape = Vec::new();
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>().map_err(GgufError::Io)? as usize);
            }
            let ggml_type = cursor.read_u32::<LittleEndian>().map_err(GgufError::Io)?;
            let offset = cursor.read_u64::<LittleEndian>().map_err(GgufError::Io)?;

            let n_elements: usize = shape.iter().product();
            let data_size = match ggml_type {
                t if t == ggml_type_GGML_TYPE_F32 as u32 => n_elements * 4,
                t if t == ggml_type_GGML_TYPE_F16 as u32 => n_elements * 2,
                t if t == ggml_type_GGML_TYPE_Q4_K as u32 => {
                    let k = 256;
                    ((n_elements + k - 1) / k) * (k / 2 + 2 * 2 + 12) // Q4_K block size
                }
                t if t == ggml_type_GGML_TYPE_Q4_0 as u32 => (n_elements / 32) * (16 + 2),
                t if t == ggml_type_GGML_TYPE_Q4_1 as u32 => (n_elements / 32) * (16 + 2 + 2),
                _ => {
                    println!("WARNING: Unknown GGML type {} for tensor {}, data_size might be wrong", ggml_type, name);
                    0
                }
            };

            tensors.insert(name.clone(), GgufTensorInfo {
                name,
                shape,
                ggml_type: ggml_type as ggml_type,
                data_offset: offset,
                data_size,
            });
        }

        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32()).unwrap_or(32) as u64;
        let pos = cursor.position();
        let data_section_offset = (pos + alignment - 1) / alignment * alignment;
        
        Ok(GgufIndex { metadata, tensors, data_section_offset, mmap })
    }

    pub fn tensor_data_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let start = (self.data_section_offset + info.data_offset) as usize;
        let end = start + info.data_size;
        Ok(&self.mmap[start..end])
    }

    pub unsafe fn load_tensor(&self, name: &str, ctx: Arc<GgmlContext>) -> Result<GgmlTensor, GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let data = self.tensor_data_bytes(name)?;

        // GGML expects exactly 4 dimensions in ne array.
        // Unused dimensions MUST be 1.
        let mut ne: [i64; 4] = [1, 1, 1, 1];
        let n_dims_raw = info.shape.len();
        let n_dims = if n_dims_raw > 4 { 4 } else { n_dims_raw };
        
        for i in 0..n_dims {
            ne[i] = info.shape[i] as i64;
        }

        println!("DEBUG: Creating tensor '{}' with ggml_type {}, n_dims {}, ne {:?}", name, info.ggml_type, n_dims, ne);

        let t = ggml_new_tensor(ctx.ptr, info.ggml_type, n_dims as i32, ne.as_ptr());
        
        if t.is_null() {
            return Err(GgufError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("ggml_new_tensor returned NULL for {}", name))));
        }

        // We need to allocate memory on the backend for this context's tensors
        ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
        
        // Move weights to backend memory
        {
            let _guard = ctx.executor.lock.lock().unwrap();
            ggml_backend_tensor_set(t, data.as_ptr() as *const std::ffi::c_void, 0, data.len());
        }

        Ok(GgmlTensor::from_raw(t, ctx))
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, std::io::Error> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_value(cursor: &mut Cursor<&[u8]>, val_type: u32) -> Result<GgufMetadataValue, std::io::Error> {
    match val_type {
        0 => Ok(GgufMetadataValue::Uint8(cursor.read_u8()?)),
        1 => Ok(GgufMetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufMetadataValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufMetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufMetadataValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufMetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufMetadataValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufMetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufMetadataValue::String(read_string(cursor)?)),
        9 => {
            let item_type = cursor.read_u32::<LittleEndian>()?;
            let len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut items = Vec::new();
            for _ in 0..len {
                items.push(read_value(cursor, item_type)?);
            }
            Ok(GgufMetadataValue::Array(items))
        }
        _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Unknown GGUF value type")),
    }
}
