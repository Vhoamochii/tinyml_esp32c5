// Bộ đọc tối giản cho file MATLAB 5.0 (.mat) của dữ liệu Paderborn.
// Hỗ trợ: phần tử nén zlib (miCOMPRESSED), mảng số, chuỗi ký tự, struct, cell.
// Không hỗ trợ: file big-endian, MATLAB v7.3 (HDF5), sparse, object (trả về Unsupported).
// Đặc tả: "MAT-File Format", MathWorks.

// Kiểu dữ liệu của phần tử (mi*)
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;
const MI_UTF32: u32 = 18;

// Lớp mảng (mx*_CLASS)
const CLASS_CELL: u8 = 1;
const CLASS_STRUCT: u8 = 2;
const CLASS_CHAR: u8 = 4;
const CLASS_DOUBLE: u8 = 6;
const CLASS_UINT64: u8 = 15;

#[derive(Debug)]
pub enum MatValue {
    Numeric { dims: Vec<usize>, data: Vec<f64> }, // Phần thực, theo thứ tự cột của MATLAB
    Char(String),
    Struct { dims: Vec<usize>, fields: Vec<String>, values: Vec<MatValue> }, // values[phần tử * số trường + trường]
    Cell { dims: Vec<usize>, values: Vec<MatValue> },
    Unsupported(u8), // Mã lớp không hỗ trợ
}

impl MatValue {
    /// Trường `field` của phần tử thứ `element` trong mảng struct
    pub fn get(&self, element: usize, field: &str) -> Option<&MatValue> {
        match self {
            MatValue::Struct { fields, values, .. } => {
                let f = fields.iter().position(|name| name == field)?;
                values.get(element * fields.len() + f)
            }
            _ => None,
        }
    }

    /// Số phần tử của mảng (struct, cell, số)
    pub fn element_count(&self) -> usize {
        match self {
            MatValue::Numeric { data, .. } => data.len(),
            MatValue::Struct { dims, .. } | MatValue::Cell { dims, .. } => dims.iter().product(),
            MatValue::Char(s) => s.chars().count(),
            MatValue::Unsupported(_) => 0,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MatValue::Char(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            MatValue::Numeric { data, .. } => Some(data),
            _ => None,
        }
    }
}

pub struct Variable {
    pub name: String,
    pub value: MatValue,
}

/// Đọc toàn bộ biến trong một file .mat (MATLAB 5.0, little-endian)
pub fn parse(bytes: &[u8]) -> Result<Vec<Variable>, String> {
    if bytes.len() < 128 {
        return Err("file qua ngan, khong phai MAT 5.0".into());
    }
    if &bytes[126..128] != b"IM" {
        return Err("chi ho tro file MAT 5.0 little-endian".into());
    }

    let mut cursor = Cursor::new(&bytes[128..]);
    let mut variables = Vec::new();
    while cursor.remaining() >= 8 {
        let (ty, data) = cursor.element()?;
        match ty {
            MI_COMPRESSED => {
                let inflated = miniz_oxide::inflate::decompress_to_vec_zlib(data)
                    .map_err(|e| format!("giai nen zlib that bai: {e}"))?;
                let mut inner = Cursor::new(&inflated);
                let (inner_ty, inner_data) = inner.element()?;
                if inner_ty == MI_MATRIX {
                    variables.push(parse_variable(inner_data)?);
                }
            }
            MI_MATRIX => variables.push(parse_variable(data)?),
            _ => {} // Bỏ qua phần tử không phải biến (ví dụ đệm)
        }
    }
    Ok(variables)
}

fn parse_variable(data: &[u8]) -> Result<Variable, String> {
    let (name, value) = parse_matrix(data)?;
    Ok(Variable { name, value })
}

/// Nội dung một phần tử miMATRIX: cờ mảng, kích thước, tên, rồi dữ liệu theo lớp
fn parse_matrix(data: &[u8]) -> Result<(String, MatValue), String> {
    if data.is_empty() {
        // Mảng rỗng có thể được ghi là miMATRIX 0 byte
        return Ok((String::new(), MatValue::Numeric { dims: vec![0, 0], data: Vec::new() }));
    }
    let mut c = Cursor::new(data);

    let (_, flags) = c.element()?;
    if flags.len() < 8 {
        return Err("co mang (array flags) khong hop le".into());
    }
    let class = flags[0];

    let (dims_ty, dims_bytes) = c.element()?;
    let dims: Vec<usize> = to_f64(dims_ty, dims_bytes)?.into_iter().map(|v| v as usize).collect();
    let count: usize = dims.iter().product();

    let (_, name_bytes) = c.element()?;
    let name = String::from_utf8_lossy(name_bytes).into_owned();

    let value = match class {
        CLASS_DOUBLE..=CLASS_UINT64 => {
            // Chỉ lấy phần thực; phần ảo (nếu có) bị bỏ qua
            let data = if c.remaining() >= 8 {
                let (ty, raw) = c.element()?;
                to_f64(ty, raw)?
            } else {
                Vec::new()
            };
            MatValue::Numeric { dims, data }
        }
        CLASS_CHAR => {
            let text = if c.remaining() >= 8 {
                let (ty, raw) = c.element()?;
                decode_chars(ty, raw)?
            } else {
                String::new()
            };
            MatValue::Char(text)
        }
        CLASS_STRUCT => {
            let (_, len_bytes) = c.element()?;
            let field_len = read_i32(len_bytes)? as usize;
            let (_, names_bytes) = c.element()?;
            let fields: Vec<String> = if field_len == 0 {
                Vec::new()
            } else {
                names_bytes
                    .chunks(field_len)
                    .map(|chunk| {
                        let end = chunk.iter().position(|&b| b == 0).unwrap_or(chunk.len());
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    })
                    .collect()
            };
            let mut values = Vec::with_capacity(count * fields.len());
            for _ in 0..count * fields.len() {
                values.push(read_child(&mut c)?);
            }
            MatValue::Struct { dims, fields, values }
        }
        CLASS_CELL => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_child(&mut c)?);
            }
            MatValue::Cell { dims, values }
        }
        other => MatValue::Unsupported(other),
    };
    Ok((name, value))
}

/// Một giá trị con (trường struct hoặc ô cell) là một phần tử miMATRIX đầy đủ
fn read_child(c: &mut Cursor<'_>) -> Result<MatValue, String> {
    let (ty, data) = c.element()?;
    if ty != MI_MATRIX {
        return Err(format!("mong doi miMATRIX trong struct/cell, gap kieu {ty}"));
    }
    Ok(parse_matrix(data)?.1)
}

fn read_i32(bytes: &[u8]) -> Result<i32, String> {
    let b: [u8; 4] = bytes.get(..4).and_then(|s| s.try_into().ok()).ok_or("thieu 4 byte cho so int32")?;
    Ok(i32::from_le_bytes(b))
}

/// Đổi dữ liệu số (mọi kiểu mi*) sang f64. MATLAB có thể lưu mảng double bằng kiểu nhỏ hơn để tiết kiệm chỗ.
fn to_f64(ty: u32, bytes: &[u8]) -> Result<Vec<f64>, String> {
    fn conv<const W: usize>(bytes: &[u8], f: impl Fn([u8; W]) -> f64) -> Vec<f64> {
        bytes.chunks_exact(W).map(|c| f(c.try_into().unwrap())).collect()
    }
    Ok(match ty {
        MI_INT8 => bytes.iter().map(|&b| b as i8 as f64).collect(),
        MI_UINT8 => bytes.iter().map(|&b| b as f64).collect(),
        MI_INT16 => conv::<2>(bytes, |b| i16::from_le_bytes(b) as f64),
        MI_UINT16 => conv::<2>(bytes, |b| u16::from_le_bytes(b) as f64),
        MI_INT32 => conv::<4>(bytes, |b| i32::from_le_bytes(b) as f64),
        MI_UINT32 => conv::<4>(bytes, |b| u32::from_le_bytes(b) as f64),
        MI_SINGLE => conv::<4>(bytes, |b| f32::from_le_bytes(b) as f64),
        MI_DOUBLE => conv::<8>(bytes, f64::from_le_bytes),
        MI_INT64 => conv::<8>(bytes, |b| i64::from_le_bytes(b) as f64),
        MI_UINT64 => conv::<8>(bytes, |b| u64::from_le_bytes(b) as f64),
        other => return Err(format!("kieu du lieu so khong ho tro: {other}")),
    })
}

fn decode_chars(ty: u32, raw: &[u8]) -> Result<String, String> {
    Ok(match ty {
        MI_UINT16 | MI_UTF16 => {
            let units: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&units)
        }
        MI_UTF8 | MI_INT8 | MI_UINT8 => String::from_utf8_lossy(raw).into_owned(),
        MI_UTF32 | MI_UINT32 | MI_INT32 => raw
            .chunks_exact(4)
            .filter_map(|c| char::from_u32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        other => return Err(format!("kieu ky tu khong ho tro: {other}")),
    })
}

/// Đọc tuần tự các phần tử dữ liệu (tag 8 byte + dữ liệu + đệm đến bội của 8)
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err(format!("du lieu bi cat ngang: can {n} byte, con {}", self.remaining()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Trả về (kiểu, dữ liệu) của phần tử tiếp theo
    fn element(&mut self) -> Result<(u32, &'a [u8]), String> {
        let first = self.u32()?;
        if first >> 16 != 0 {
            // Phần tử nhỏ: kiểu và số byte gói trong 4 byte đầu, dữ liệu nằm trong 4 byte sau
            let ty = first & 0xFFFF;
            let n = (first >> 16) as usize;
            if n > 4 {
                return Err(format!("phan tu nho co {n} byte (> 4)"));
            }
            let data = self.take(4)?;
            return Ok((ty, &data[..n]));
        }
        let ty = first;
        let n = self.u32()? as usize;
        let data = self.take(n)?;
        if ty != MI_COMPRESSED {
            // Phần tử thường được đệm đến bội của 8 byte; phần tử nén thì không
            let pad = (8 - n % 8) % 8;
            self.pos += pad.min(self.remaining());
        }
        Ok((ty, data))
    }
}
