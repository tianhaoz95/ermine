use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;
use byteorder::{ReadBytesExt, LittleEndian};
use std::io::{Cursor, Read};
use std::sync::Arc;
use crate::context::GgmlContext;
use crate::tensor::GgmlTensor;
use ggml_sys::*;
use std::ffi::c_void;

/// Metadata about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    /// Byte offset of tensor data from the start of the tensor data section.
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
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
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Parsed GGUF index: metadata + tensor directory, no tensor data loaded yet.
pub struct GgufIndex {
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Byte offset in the file where tensor data begins.
    pub data_section_offset: u64,
    mmap: Mmap,
}

impl GgufIndex {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(mmap.as_ref());
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != 0x46554747 {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Invalid GGUF magic"));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported GGUF version"));
        }

        let n_tensors = cursor.read_u64::<LittleEndian>()?;
        let n_kv = cursor.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut cursor)?;
            let val_type = cursor.read_u32::<LittleEndian>()?;
            let val = read_value(&mut cursor, val_type)?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let ggml_type = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            tensors.insert(name.clone(), GgufTensorInfo {
                name,
                shape,
                ggml_type,
                data_offset: offset,
            });
        }

        let alignment = metadata.get("general.alignment")
            .and_then(|v| match v {
                GgufMetadataValue::Uint32(u) => Some(*u as u64),
                GgufMetadataValue::Uint64(u) => Some(*u),
                _ => None
            }).unwrap_or(32);

        let pos = cursor.position();
        let data_section_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufIndex { metadata, tensors, data_section_offset, mmap })
    }

    pub fn get_layer_tensors(&self, layer_idx: usize) -> Vec<String> {
        let prefix = format!("blk.{}.", layer_idx);
        self.tensors.keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect()
    }

    pub fn get_expert_tensors(&self, layer_idx: usize, expert_idx: usize) -> Vec<String> {
        // llama.cpp naming for MoE: blk.N.ffn_gate.M.weight etc? 
        // Need to check actual naming. For Gemma 4 MoE it might be different.
        // Assuming blk.N.ffn_expert.M prefix for now.
        let prefix = format!("blk.{}.ffn_expert.{}.", layer_idx, expert_idx);
        self.tensors.keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect()
    }

    pub unsafe fn load_tensor(&self, name: &str, ctx: &Arc<GgmlContext>) -> Result<GgmlTensor, String> {
        let info = self.tensors.get(name).ok_or_else(|| format!("Tensor not found: {}", name))?;
        
        let mut dims = [1i64; 4];
        for (i, &d) in info.shape.iter().enumerate() {
            dims[i] = d as i64;
        }

        let t = ggml_new_tensor(ctx.ptr, (info.ggml_type as i32).try_into().unwrap(), info.shape.len() as i32, dims.as_ptr());
        
        // Find data size
        let n_elements = info.shape.iter().product::<usize>();
        let type_size = ggml_type_size((info.ggml_type as i32).try_into().unwrap());
        let blck_size = ggml_blck_size((info.ggml_type as i32).try_into().unwrap()) as usize;
        let data_size = (n_elements * type_size) / blck_size;

        let start = (self.data_section_offset + info.data_offset) as usize;
        let end = start + data_size;
        let data = &self.mmap[start..end];

        // Allocate on backend
        let executor = ctx.executor.clone(); let _guard = executor.lock.lock().unwrap();
        ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
        ggml_backend_tensor_set(t, data.as_ptr() as *const c_void, 0, data.len());

        Ok(GgmlTensor::from_raw(t, ctx.clone()))
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, std::io::Error> {
    let len = cursor.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
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
            let len = cursor.read_u64::<LittleEndian>()?;
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(read_value(cursor, item_type)?);
            }
            Ok(GgufMetadataValue::Array(items))
        }
        10 => Ok(GgufMetadataValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufMetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufMetadataValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Unknown GGUF value type: {}", val_type))),
    }
}
