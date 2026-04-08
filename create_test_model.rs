use std::io::{Write, Cursor};
use byteorder::{WriteBytesExt, LittleEndian};

fn main() {
    let mut buf = Vec::new();
    
    // Magic and Version
    buf.write_u32::<LittleEndian>(0x46554747).unwrap(); // 'GGUF'
    buf.write_u32::<LittleEndian>(3).unwrap();
    
    // Header: 1 tensor, 1 KV pair
    buf.write_u64::<LittleEndian>(1).unwrap();
    buf.write_u64::<LittleEndian>(1).unwrap();
    
    // Metadata: general.alignment = 32
    write_string(&mut buf, "general.alignment");
    buf.write_u32::<LittleEndian>(4).unwrap(); // Uint32
    buf.write_u32::<LittleEndian>(32).unwrap();
    
    // Tensor Info: "test.weight", [2, 2], F32, offset 0
    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(2).unwrap(); // d1
    buf.write_u64::<LittleEndian>(2).unwrap(); // d2
    buf.write_u32::<LittleEndian>(0).unwrap(); // GGML_TYPE_F32
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset
    
    // Padding to 32 bytes
    let pos = buf.len();
    let pad = (32 - (pos % 32)) % 32;
    for _ in 0..pad { buf.write_u8(0).unwrap(); }
    
    // Tensor Data: [[1.0, 2.0], [3.0, 4.0]]
    buf.write_f32::<LittleEndian>(1.0).unwrap();
    buf.write_f32::<LittleEndian>(2.0).unwrap();
    buf.write_f32::<LittleEndian>(3.0).unwrap();
    buf.write_f32::<LittleEndian>(4.0).unwrap();
    
    std::fs::write("test_model.gguf", buf).unwrap();
    println!("Created test_model.gguf");
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}
