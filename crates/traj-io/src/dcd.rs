use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

#[derive(Debug, Clone, Copy)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug)]
pub struct DcdReader {
    file: BufReader<File>,
    endian: Endian,
    marker_size: usize,
    n_atoms: usize,
    n_frames: Option<usize>,
    data_start: u64,
    unitcell_layout: UnitCellLayout,
    length_scale: f32,
    axis_buf: Vec<u8>,
    axis_f32: Vec<f32>,
    axis_f32_b: Vec<f32>,
    axis_f32_c: Vec<f32>,
    selection_cache_u32: Vec<u32>,
    selection_cache_usize: Vec<usize>,
}

const DCD_IO_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
enum UnitCellLayout {
    StandardAbcAngles,
    CharmmAgBcAngles,
}

impl DcdReader {
    pub fn open(path: impl Into<PathBuf>, length_scale: f32) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let mut file = BufReader::with_capacity(DCD_IO_BUFFER_BYTES, file);
        let (endian, marker_size, header_len) = detect_header_marker(&mut file)?;
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| TrajError::Parse("DCD header length too large".into()))?;
        let mut header = vec![0u8; header_len_usize];
        file.read_exact(&mut header)?;
        let trailer = read_marker(&mut file, endian, marker_size)?;
        if trailer != header_len {
            return Err(TrajError::Parse("DCD header record length mismatch".into()));
        }
        let n_frames = parse_n_frames(&header, endian);
        let unitcell_layout = parse_unitcell_layout(&header, endian);

        // Skip title record
        skip_record(&mut file, endian, marker_size)?;

        // Read natoms record
        let natoms_len = read_marker(&mut file, endian, marker_size)?;
        if natoms_len != 4 {
            return Err(TrajError::Parse("unexpected natoms record length".into()));
        }
        let natoms = read_i32(&mut file, endian)?;
        let natoms_end = read_marker(&mut file, endian, marker_size)?;
        if natoms_end != natoms_len {
            return Err(TrajError::Parse("natoms record length mismatch".into()));
        }
        if natoms <= 0 {
            return Err(TrajError::Parse("invalid natoms".into()));
        }
        let data_start = file.stream_position()?;

        Ok(Self {
            file,
            endian,
            marker_size,
            n_atoms: natoms as usize,
            n_frames,
            data_start,
            unitcell_layout,
            length_scale,
            axis_buf: Vec::with_capacity((natoms as usize).saturating_mul(4)),
            axis_f32: Vec::with_capacity(natoms as usize),
            axis_f32_b: Vec::with_capacity(natoms as usize),
            axis_f32_c: Vec::with_capacity(natoms as usize),
            selection_cache_u32: Vec::new(),
            selection_cache_usize: Vec::new(),
        })
    }

    fn ensure_selection_cache(&mut self, selection: &[u32]) -> TrajResult<()> {
        if self.selection_cache_u32.as_slice() == selection {
            return Ok(());
        }
        self.selection_cache_usize = validate_and_materialize_selection(selection, self.n_atoms)?;
        self.selection_cache_u32.clear();
        self.selection_cache_u32.extend_from_slice(selection);
        Ok(())
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        self.file.seek(SeekFrom::Start(self.data_start))?;
        Ok(())
    }
}

pub struct DcdWriter {
    file: BufWriter<File>,
    n_atoms: usize,
}

impl TrajReader for DcdReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        self.n_frames
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        out.reset(self.n_atoms, max_frames);
        let mut frames = 0;
        while frames < max_frames {
            match self.read_frame(out)? {
                true => frames += 1,
                false => break,
            }
        }
        Ok(frames)
    }

    fn read_chunk_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        out: &mut FrameChunkBuilder,
    ) -> TrajResult<usize> {
        self.ensure_selection_cache(selection)?;
        out.reset(self.selection_cache_usize.len(), max_frames);
        let mut frames = 0;
        while frames < max_frames {
            match self.read_frame_selected_cached(out)? {
                true => frames += 1,
                false => break,
            }
        }
        Ok(frames)
    }

    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let mut skipped = 0usize;
        while skipped < n_frames {
            match self.skip_frame()? {
                true => skipped += 1,
                false => break,
            }
        }
        Ok(skipped)
    }
}

impl DcdReader {
    fn skip_frame(&mut self) -> TrajResult<bool> {
        let expected_len = (self.n_atoms * 4) as u64;
        let mut len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
            Some(l) => l,
            None => return Ok(false),
        };

        if len != expected_len && is_unitcell_len(len) {
            skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
            len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
                Some(l) => l,
                None => return Ok(false),
            };
        }
        if len != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
        let len_b = read_marker(&mut self.file, self.endian, self.marker_size)?;
        if len_b != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        skip_record_with_len(&mut self.file, self.endian, self.marker_size, len_b)?;
        let len_c = read_marker(&mut self.file, self.endian, self.marker_size)?;
        if len_c != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        skip_record_with_len(&mut self.file, self.endian, self.marker_size, len_c)?;
        Ok(true)
    }

    fn read_frame(&mut self, out: &mut FrameChunkBuilder) -> TrajResult<bool> {
        let expected_len = (self.n_atoms * 4) as u64;
        let needs_box = out.needs_box();
        let mut len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
            Some(l) => l,
            None => return Ok(false),
        };

        let mut box_ = Box3::None;
        if len != expected_len {
            if is_unitcell_len(len) {
                box_ = if needs_box {
                    read_unitcell_with_len(
                        &mut self.file,
                        self.endian,
                        self.marker_size,
                        len,
                        self.unitcell_layout,
                        self.length_scale,
                    )?
                } else {
                    skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
                    Box3::None
                };
                len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
                    Some(l) => l,
                    None => return Ok(false),
                };
            }
        }

        if len != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len,
            &mut self.axis_buf,
            &mut self.axis_f32,
        )?;
        let len_b = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_b,
            &mut self.axis_buf,
            &mut self.axis_f32_b,
        )?;
        let len_c = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_c,
            &mut self.axis_buf,
            &mut self.axis_f32_c,
        )?;

        let coords = out.start_frame(box_, None);
        if self.length_scale == 1.0 {
            for (i, dst) in coords.iter_mut().enumerate() {
                dst[0] = self.axis_f32[i];
                dst[1] = self.axis_f32_b[i];
                dst[2] = self.axis_f32_c[i];
                dst[3] = 1.0;
            }
        } else {
            let scale = self.length_scale;
            for (i, dst) in coords.iter_mut().enumerate() {
                dst[0] = self.axis_f32[i] * scale;
                dst[1] = self.axis_f32_b[i] * scale;
                dst[2] = self.axis_f32_c[i] * scale;
                dst[3] = 1.0;
            }
        }
        Ok(true)
    }

    fn read_frame_selected_cached(&mut self, out: &mut FrameChunkBuilder) -> TrajResult<bool> {
        let expected_len = (self.n_atoms * 4) as u64;
        let needs_box = out.needs_box();
        let mut len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
            Some(l) => l,
            None => return Ok(false),
        };

        let mut box_ = Box3::None;
        if len != expected_len {
            if is_unitcell_len(len) {
                box_ = if needs_box {
                    read_unitcell_with_len(
                        &mut self.file,
                        self.endian,
                        self.marker_size,
                        len,
                        self.unitcell_layout,
                        self.length_scale,
                    )?
                } else {
                    skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
                    Box3::None
                };
                len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
                    Some(l) => l,
                    None => return Ok(false),
                };
            }
        }

        if len != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len,
            &mut self.axis_buf,
            &mut self.axis_f32,
        )?;
        let len_b = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_b,
            &mut self.axis_buf,
            &mut self.axis_f32_b,
        )?;
        let len_c = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_c,
            &mut self.axis_buf,
            &mut self.axis_f32_c,
        )?;

        let coords = out.start_frame(box_, None);
        let selection = self.selection_cache_usize.as_slice();
        if self.length_scale == 1.0 {
            fill_selected_axes_unit(
                coords,
                &self.axis_f32,
                &self.axis_f32_b,
                &self.axis_f32_c,
                selection,
            );
        } else {
            let scale = self.length_scale;
            fill_selected_axes_scaled(
                coords,
                &self.axis_f32,
                &self.axis_f32_b,
                &self.axis_f32_c,
                selection,
                scale,
            );
        }
        Ok(true)
    }

    pub fn read_chunk_into_coords3(
        &mut self,
        max_frames: usize,
        coords_out: &mut [f32],
    ) -> TrajResult<usize> {
        let frames_cap = coords_out.len() / (self.n_atoms * 3);
        if frames_cap == 0 {
            return Ok(0);
        }
        let target = max_frames.min(frames_cap).max(1);
        let mut frames = 0usize;
        while frames < target {
            let start = frames * self.n_atoms * 3;
            let end = start + self.n_atoms * 3;
            match self.read_frame_into_coords3(&mut coords_out[start..end])? {
                true => frames += 1,
                false => break,
            }
        }
        Ok(frames)
    }

    pub fn read_chunk_into_coords3_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        coords_out: &mut [f32],
    ) -> TrajResult<usize> {
        self.ensure_selection_cache(selection)?;
        let n_sel = self.selection_cache_usize.len();
        if n_sel == 0 {
            return Ok(0);
        }
        let frames_cap = coords_out.len() / (n_sel * 3);
        if frames_cap == 0 {
            return Ok(0);
        }
        let target = max_frames.min(frames_cap).max(1);
        let mut frames = 0usize;
        while frames < target {
            let start = frames * n_sel * 3;
            let end = start + n_sel * 3;
            match self.read_frame_selected_into_coords3(&mut coords_out[start..end])? {
                true => frames += 1,
                false => break,
            }
        }
        Ok(frames)
    }

    fn read_frame_into_coords3(&mut self, coords_out: &mut [f32]) -> TrajResult<bool> {
        let expected_len = (self.n_atoms * 4) as u64;
        let mut len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
            Some(l) => l,
            None => return Ok(false),
        };
        if len != expected_len && is_unitcell_len(len) {
            skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
            len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
                Some(l) => l,
                None => return Ok(false),
            };
        }
        if len != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len,
            &mut self.axis_buf,
            &mut self.axis_f32,
        )?;
        let len_b = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_b,
            &mut self.axis_buf,
            &mut self.axis_f32_b,
        )?;
        let len_c = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_c,
            &mut self.axis_buf,
            &mut self.axis_f32_c,
        )?;

        if self.length_scale == 1.0 {
            for i in 0..self.n_atoms {
                let base = i * 3;
                coords_out[base] = self.axis_f32[i];
                coords_out[base + 1] = self.axis_f32_b[i];
                coords_out[base + 2] = self.axis_f32_c[i];
            }
        } else {
            let scale = self.length_scale;
            for i in 0..self.n_atoms {
                let base = i * 3;
                coords_out[base] = self.axis_f32[i] * scale;
                coords_out[base + 1] = self.axis_f32_b[i] * scale;
                coords_out[base + 2] = self.axis_f32_c[i] * scale;
            }
        }
        Ok(true)
    }

    fn read_frame_selected_into_coords3(&mut self, coords_out: &mut [f32]) -> TrajResult<bool> {
        let expected_len = (self.n_atoms * 4) as u64;
        let mut len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
            Some(l) => l,
            None => return Ok(false),
        };
        if len != expected_len && is_unitcell_len(len) {
            skip_record_with_len(&mut self.file, self.endian, self.marker_size, len)?;
            len = match read_marker_opt(&mut self.file, self.endian, self.marker_size)? {
                Some(l) => l,
                None => return Ok(false),
            };
        }
        if len != expected_len {
            return Err(TrajError::Parse(
                "unexpected DCD coordinate record length".into(),
            ));
        }
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len,
            &mut self.axis_buf,
            &mut self.axis_f32,
        )?;
        let len_b = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_b,
            &mut self.axis_buf,
            &mut self.axis_f32_b,
        )?;
        let len_c = read_marker(&mut self.file, self.endian, self.marker_size)?;
        read_axis_payload_into_buffer(
            &mut self.file,
            self.endian,
            self.marker_size,
            self.n_atoms,
            len_c,
            &mut self.axis_buf,
            &mut self.axis_f32_c,
        )?;

        let selection = self.selection_cache_usize.as_slice();
        if self.length_scale == 1.0 {
            fill_selected_axes3_unit(
                coords_out,
                &self.axis_f32,
                &self.axis_f32_b,
                &self.axis_f32_c,
                selection,
            );
        } else {
            let scale = self.length_scale;
            fill_selected_axes3_scaled(
                coords_out,
                &self.axis_f32,
                &self.axis_f32_b,
                &self.axis_f32_c,
                selection,
                scale,
            );
        }
        Ok(true)
    }
}

impl DcdWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize, n_frames: usize) -> TrajResult<Self> {
        let path = path.into();
        let file = File::create(path)?;
        let mut file = BufWriter::with_capacity(DCD_IO_BUFFER_BYTES, file);
        write_dcd_header(&mut file, n_atoms, n_frames)?;
        Ok(Self { file, n_atoms })
    }

    pub fn write_frame(&mut self, coords: &[[f32; 3]], box_: Box3) -> TrajResult<()> {
        if coords.len() != self.n_atoms {
            return Err(TrajError::Mismatch(format!(
                "frame atom count {} does not match writer atom count {}",
                coords.len(),
                self.n_atoms
            )));
        }
        write_dcd_record(&mut self.file, &box_to_dcd_payload(box_))?;
        let mut x = Vec::with_capacity(self.n_atoms);
        let mut y = Vec::with_capacity(self.n_atoms);
        let mut z = Vec::with_capacity(self.n_atoms);
        for atom in coords.iter() {
            x.push(atom[0]);
            y.push(atom[1]);
            z.push(atom[2]);
        }
        write_dcd_f32_record(&mut self.file, &x)?;
        write_dcd_f32_record(&mut self.file, &y)?;
        write_dcd_f32_record(&mut self.file, &z)?;
        Ok(())
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        self.file.flush()?;
        Ok(())
    }
}

fn read_axis_payload_into_buffer(
    file: &mut impl Read,
    endian: Endian,
    marker_size: usize,
    count: usize,
    len: u64,
    axis_buf: &mut Vec<u8>,
    axis_f32: &mut Vec<f32>,
) -> TrajResult<()> {
    let expected_len = (count * 4) as u64;
    if len != expected_len {
        return Err(TrajError::Parse("unexpected float record length".into()));
    }
    if little_endian_fast_path(endian) {
        if axis_f32.len() < count {
            axis_f32.resize(count, 0.0);
        }
        let dst = bytemuck::cast_slice_mut(&mut axis_f32[..count]);
        file.read_exact(dst)?;
    } else {
        let need = count * 4;
        if axis_buf.len() < need {
            axis_buf.resize(need, 0);
        }
        file.read_exact(&mut axis_buf[..need])?;
        if axis_f32.len() < count {
            axis_f32.resize(count, 0.0);
        }
        let bytes = &axis_buf[..need];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            axis_f32[i] = match endian {
                Endian::Little => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                Endian::Big => f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            };
        }
    }
    let end_len = read_marker(file, endian, marker_size)?;
    if end_len != len {
        return Err(TrajError::Parse("float record length mismatch".into()));
    }
    Ok(())
}

fn write_dcd_header(file: &mut impl Write, n_atoms: usize, n_frames: usize) -> TrajResult<()> {
    let mut header = Vec::with_capacity(84);
    header.extend_from_slice(b"CORD");
    let mut icntrl = [0i32; 20];
    icntrl[0] = i32::try_from(n_frames)
        .map_err(|_| TrajError::Invalid("too many frames for DCD header".into()))?;
    icntrl[10] = 1;
    for value in icntrl {
        header.extend_from_slice(&value.to_le_bytes());
    }
    write_dcd_record(file, &header)?;

    let mut title = Vec::with_capacity(84);
    title.extend_from_slice(&1i32.to_le_bytes());
    let mut line = [b' '; 80];
    let text = b"WARP_MD_DCD";
    line[..text.len()].copy_from_slice(text);
    title.extend_from_slice(&line);
    write_dcd_record(file, &title)?;

    let natoms = i32::try_from(n_atoms)
        .map_err(|_| TrajError::Invalid("too many atoms for DCD header".into()))?;
    write_dcd_record(file, &natoms.to_le_bytes())?;
    Ok(())
}

fn write_dcd_record(file: &mut impl Write, payload: &[u8]) -> TrajResult<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| TrajError::Invalid("DCD record payload too large".into()))?;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(payload)?;
    file.write_all(&len.to_le_bytes())?;
    Ok(())
}

fn write_dcd_f32_record(file: &mut impl Write, values: &[f32]) -> TrajResult<()> {
    let mut payload = Vec::with_capacity(values.len() * 4);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    write_dcd_record(file, &payload)
}

fn box_to_dcd_payload(box_: Box3) -> [u8; 48] {
    let values = box_to_dcd_unitcell_values(box_);
    let mut payload = [0u8; 48];
    for (i, value) in values.iter().enumerate() {
        let start = i * 8;
        payload[start..start + 8].copy_from_slice(&value.to_le_bytes());
    }
    payload
}

fn box_to_dcd_unitcell_values(box_: Box3) -> [f64; 6] {
    let (a, b, c, alpha, beta, gamma) = box_to_lengths_and_angles(box_);
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return [0.0; 6];
    }
    [
        a as f64,
        (gamma.to_radians().cos()) as f64,
        b as f64,
        (beta.to_radians().cos()) as f64,
        (alpha.to_radians().cos()) as f64,
        c as f64,
    ]
}

fn box_to_lengths_and_angles(box_: Box3) -> (f32, f32, f32, f32, f32, f32) {
    match box_ {
        Box3::None => (0.0, 0.0, 0.0, 90.0, 90.0, 90.0),
        Box3::Orthorhombic { lx, ly, lz } => (lx, ly, lz, 90.0, 90.0, 90.0),
        Box3::Triclinic { m } => {
            let a_vec = [m[0], m[1], m[2]];
            let b_vec = [m[3], m[4], m[5]];
            let c_vec = [m[6], m[7], m[8]];
            (
                vec_len(a_vec),
                vec_len(b_vec),
                vec_len(c_vec),
                angle_deg(b_vec, c_vec),
                angle_deg(a_vec, c_vec),
                angle_deg(a_vec, b_vec),
            )
        }
    }
}

fn vec_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
    let denom = vec_len(a) * vec_len(b);
    if denom <= 0.0 {
        return 90.0;
    }
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cosine = (dot / denom).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

#[inline]
fn little_endian_fast_path(endian: Endian) -> bool {
    cfg!(target_endian = "little") && matches!(endian, Endian::Little)
}

#[inline(always)]
fn fill_selected_axes_unit(
    dst: &mut [[f32; 4]],
    x: &[f32],
    y: &[f32],
    z: &[f32],
    selection: &[usize],
) {
    // Bounds are validated once per chunk before entering this hot loop.
    for i in 0..selection.len() {
        let src = unsafe { *selection.get_unchecked(i) };
        let out = unsafe { dst.get_unchecked_mut(i) };
        out[0] = unsafe { *x.get_unchecked(src) };
        out[1] = unsafe { *y.get_unchecked(src) };
        out[2] = unsafe { *z.get_unchecked(src) };
        out[3] = 1.0;
    }
}

#[inline(always)]
fn fill_selected_axes_scaled(
    dst: &mut [[f32; 4]],
    x: &[f32],
    y: &[f32],
    z: &[f32],
    selection: &[usize],
    scale: f32,
) {
    // Bounds are validated once per chunk before entering this hot loop.
    for i in 0..selection.len() {
        let src = unsafe { *selection.get_unchecked(i) };
        let out = unsafe { dst.get_unchecked_mut(i) };
        out[0] = unsafe { *x.get_unchecked(src) } * scale;
        out[1] = unsafe { *y.get_unchecked(src) } * scale;
        out[2] = unsafe { *z.get_unchecked(src) } * scale;
        out[3] = 1.0;
    }
}

#[inline(always)]
fn fill_selected_axes3_unit(dst: &mut [f32], x: &[f32], y: &[f32], z: &[f32], selection: &[usize]) {
    // Bounds are validated once per chunk before entering this hot loop.
    for i in 0..selection.len() {
        let src = unsafe { *selection.get_unchecked(i) };
        let out = i * 3;
        dst[out] = unsafe { *x.get_unchecked(src) };
        dst[out + 1] = unsafe { *y.get_unchecked(src) };
        dst[out + 2] = unsafe { *z.get_unchecked(src) };
    }
}

#[inline(always)]
fn fill_selected_axes3_scaled(
    dst: &mut [f32],
    x: &[f32],
    y: &[f32],
    z: &[f32],
    selection: &[usize],
    scale: f32,
) {
    // Bounds are validated once per chunk before entering this hot loop.
    for i in 0..selection.len() {
        let src = unsafe { *selection.get_unchecked(i) };
        let out = i * 3;
        dst[out] = unsafe { *x.get_unchecked(src) } * scale;
        dst[out + 1] = unsafe { *y.get_unchecked(src) } * scale;
        dst[out + 2] = unsafe { *z.get_unchecked(src) } * scale;
    }
}

fn validate_and_materialize_selection(selection: &[u32], n_atoms: usize) -> TrajResult<Vec<usize>> {
    let mut out = Vec::with_capacity(selection.len());
    for &idx in selection {
        let src = idx as usize;
        if src >= n_atoms {
            return Err(TrajError::Mismatch(format!(
                "selection index {idx} out of bounds for trajectory with {n_atoms} atoms"
            )));
        }
        out.push(src);
    }
    Ok(out)
}

fn detect_header_marker(file: &mut (impl Read + Seek)) -> TrajResult<(Endian, usize, u64)> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    let len64_le = u64::from_le_bytes(buf);
    let len64_be = u64::from_be_bytes(buf);
    if is_header_len_u64(len64_le) {
        return Ok((Endian::Little, 8, len64_le));
    }
    if is_header_len_u64(len64_be) {
        return Ok((Endian::Big, 8, len64_be));
    }
    file.seek(SeekFrom::Start(0))?;
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let len_le = u32::from_le_bytes(buf4);
    let len_be = u32::from_be_bytes(buf4);
    if is_header_len(len_le) {
        Ok((Endian::Little, 4, len_le as u64))
    } else if is_header_len(len_be) {
        Ok((Endian::Big, 4, len_be as u64))
    } else {
        Err(TrajError::Unsupported(
            "unsupported DCD record marker".into(),
        ))
    }
}

fn is_header_len(len: u32) -> bool {
    matches!(len, 84 | 164)
}

fn is_header_len_u64(len: u64) -> bool {
    matches!(len, 84 | 164)
}

fn is_unitcell_len(len: u64) -> bool {
    matches!(len, 48 | 24)
}

fn read_marker(file: &mut impl Read, endian: Endian, marker_size: usize) -> TrajResult<u64> {
    match marker_size {
        4 => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            Ok(match endian {
                Endian::Little => u32::from_le_bytes(buf) as u64,
                Endian::Big => u32::from_be_bytes(buf) as u64,
            })
        }
        8 => {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            Ok(match endian {
                Endian::Little => u64::from_le_bytes(buf),
                Endian::Big => u64::from_be_bytes(buf),
            })
        }
        _ => Err(TrajError::Unsupported("unsupported DCD marker size".into())),
    }
}

fn read_marker_opt(
    file: &mut impl Read,
    endian: Endian,
    marker_size: usize,
) -> TrajResult<Option<u64>> {
    match marker_size {
        4 => {
            let mut buf = [0u8; 4];
            match file.read_exact(&mut buf) {
                Ok(()) => Ok(Some(match endian {
                    Endian::Little => u32::from_le_bytes(buf) as u64,
                    Endian::Big => u32::from_be_bytes(buf) as u64,
                })),
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                Err(err) => Err(err.into()),
            }
        }
        8 => {
            let mut buf = [0u8; 8];
            match file.read_exact(&mut buf) {
                Ok(()) => Ok(Some(match endian {
                    Endian::Little => u64::from_le_bytes(buf),
                    Endian::Big => u64::from_be_bytes(buf),
                })),
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                Err(err) => Err(err.into()),
            }
        }
        _ => Err(TrajError::Unsupported("unsupported DCD marker size".into())),
    }
}

fn read_i32(file: &mut impl Read, endian: Endian) -> TrajResult<i32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(buf),
        Endian::Big => i32::from_be_bytes(buf),
    })
}

fn read_f32(file: &mut impl Read, endian: Endian) -> TrajResult<f32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(buf),
        Endian::Big => f32::from_be_bytes(buf),
    })
}

fn read_f64(file: &mut impl Read, endian: Endian) -> TrajResult<f64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(match endian {
        Endian::Little => f64::from_le_bytes(buf),
        Endian::Big => f64::from_be_bytes(buf),
    })
}

fn skip_record(file: &mut impl Read, endian: Endian, marker_size: usize) -> TrajResult<()> {
    let len = read_marker(file, endian, marker_size)?;
    skip_record_with_len(file, endian, marker_size, len)
}

fn skip_record_with_len(
    file: &mut impl Read,
    endian: Endian,
    marker_size: usize,
    len: u64,
) -> TrajResult<()> {
    let mut remain = len as usize;
    let mut scratch = [0u8; 256];
    while remain > 0 {
        let take = remain.min(scratch.len());
        file.read_exact(&mut scratch[..take])?;
        remain -= take;
    }
    let end_len = read_marker(file, endian, marker_size)?;
    if end_len != len {
        return Err(TrajError::Parse("record length mismatch".into()));
    }
    Ok(())
}

fn read_unitcell_with_len(
    file: &mut impl Read,
    endian: Endian,
    marker_size: usize,
    len: u64,
    layout: UnitCellLayout,
    length_scale: f32,
) -> TrajResult<Box3> {
    let mut values = [0.0f64; 6];
    match len {
        24 => {
            for i in 0..6 {
                values[i] = read_f32(file, endian)? as f64;
            }
        }
        48 => {
            for i in 0..6 {
                values[i] = read_f64(file, endian)?;
            }
        }
        _ => {
            skip_record_with_len(file, endian, marker_size, len)?;
            return Ok(Box3::None);
        }
    }
    let end_len = read_marker(file, endian, marker_size)?;
    if end_len != len {
        return Err(TrajError::Parse("unitcell record length mismatch".into()));
    }

    let primary = box_from_values(values, layout, length_scale);
    if !matches!(primary, Box3::None) {
        return Ok(primary);
    }
    // Fallback with alternate layout for resilience across producer variants.
    let alternate = box_from_values(
        values,
        match layout {
            UnitCellLayout::StandardAbcAngles => UnitCellLayout::CharmmAgBcAngles,
            UnitCellLayout::CharmmAgBcAngles => UnitCellLayout::StandardAbcAngles,
        },
        length_scale,
    );
    Ok(alternate)
}

fn box_from_values(values: [f64; 6], layout: UnitCellLayout, length_scale: f32) -> Box3 {
    let (mut a, mut b, mut c, alpha_raw, beta_raw, gamma_raw) = match layout {
        UnitCellLayout::StandardAbcAngles => (
            values[0], values[1], values[2], values[3], values[4], values[5],
        ),
        // CHARMM/OpenMM order:
        // [A, gamma, B, beta, alpha, C], where angles can be cosines or degrees.
        UnitCellLayout::CharmmAgBcAngles => (
            values[0], values[2], values[5], values[4], values[3], values[1],
        ),
    };
    a *= length_scale as f64;
    b *= length_scale as f64;
    c *= length_scale as f64;
    let min_len = 1e-6;
    if !a.is_finite()
        || !b.is_finite()
        || !c.is_finite()
        || a <= min_len
        || b <= min_len
        || c <= min_len
    {
        return Box3::None;
    }
    let alpha_rad = unitcell_angle_to_radians(alpha_raw);
    let beta_rad = unitcell_angle_to_radians(beta_raw);
    let gamma_rad = unitcell_angle_to_radians(gamma_raw);
    if !alpha_rad.is_finite() || !beta_rad.is_finite() || !gamma_rad.is_finite() {
        return Box3::None;
    }
    let ninety = std::f64::consts::FRAC_PI_2;
    let tol = 1e-3;
    if (alpha_rad - ninety).abs() < tol
        && (beta_rad - ninety).abs() < tol
        && (gamma_rad - ninety).abs() < tol
    {
        return Box3::Orthorhombic {
            lx: a as f32,
            ly: b as f32,
            lz: c as f32,
        };
    }

    let cos_alpha = alpha_rad.cos();
    let cos_beta = beta_rad.cos();
    let cos_gamma = gamma_rad.cos();
    let sin_gamma = gamma_rad.sin();
    if sin_gamma.abs() < 1e-8 {
        return Box3::None;
    }
    let ax = a;
    let ay = 0.0;
    let az = 0.0;
    let bx = b * cos_gamma;
    let by = b * sin_gamma;
    let bz = 0.0;
    let cx = c * cos_beta;
    let cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    let cz_sq = c * c - cx * cx - cy * cy;
    if !cz_sq.is_finite() || cz_sq <= 0.0 {
        return Box3::None;
    }
    let cz = cz_sq.sqrt();
    Box3::Triclinic {
        m: [
            ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, cx as f32, cy as f32,
            cz as f32,
        ],
    }
}

fn unitcell_angle_to_radians(value: f64) -> f64 {
    if value.abs() <= 1.0 {
        value.clamp(-1.0, 1.0).acos()
    } else {
        value.to_radians()
    }
}

fn parse_n_frames(header: &[u8], endian: Endian) -> Option<usize> {
    if header.len() < 8 {
        return None;
    }
    if &header[0..4] != b"CORD" {
        return None;
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&header[4..8]);
    let n = match endian {
        Endian::Little => i32::from_le_bytes(raw),
        Endian::Big => i32::from_be_bytes(raw),
    };
    (n > 0).then_some(n as usize)
}

fn parse_unitcell_layout(header: &[u8], endian: Endian) -> UnitCellLayout {
    // CHARMM DCD writes icntrl[19] (version) > 0 and uses [A, gamma, B, beta, alpha, C].
    if header.len() >= 84 {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&header[80..84]);
        let version = match endian {
            Endian::Little => i32::from_le_bytes(raw),
            Endian::Big => i32::from_be_bytes(raw),
        };
        if version > 0 {
            return UnitCellLayout::CharmmAgBcAngles;
        }
    }
    UnitCellLayout::StandardAbcAngles
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    #[test]
    fn reject_invalid_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.dcd");
        let mut file = File::create(&path).unwrap();
        file.write_all(&[1, 2, 3, 4]).unwrap();
        let err = DcdReader::open(&path, 1.0).unwrap_err();
        match err {
            TrajError::Unsupported(_) | TrajError::Io(_) | TrajError::Parse(_) => {}
            _ => panic!("unexpected error"),
        }
    }

    #[test]
    fn read_frame_without_unitcell_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_box.dcd");
        write_test_dcd(&path, 2, None, &[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0], false);

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 1);
        let frames = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(frames, 1);
        assert_eq!(reader.n_frames_hint(), Some(1));
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.box_[0], Box3::None);
        assert_eq!(chunk.coords[0], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(chunk.coords[1], [4.0, 5.0, 6.0, 1.0]);
    }

    #[test]
    fn read_frame_selected_endpoints() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("selected_no_box.dcd");
        write_test_dcd(
            &path,
            4,
            None,
            &[1.0, 2.0, 3.0, 4.0],
            &[10.0, 20.0, 30.0, 40.0],
            &[100.0, 200.0, 300.0, 400.0],
            false,
        );

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 1);
        let frames = reader
            .read_chunk_selected(1, &[3, 1], &mut builder)
            .unwrap();
        assert_eq!(frames, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.n_atoms, 2);
        assert_eq!(chunk.coords[0], [4.0, 40.0, 400.0, 1.0]);
        assert_eq!(chunk.coords[1], [2.0, 20.0, 200.0, 1.0]);
    }

    #[test]
    fn reset_rewinds_dcd_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reset_no_box.dcd");
        write_test_dcd(&path, 1, None, &[1.0], &[2.0], &[3.0], false);

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut out = vec![0.0f32; 3];
        let read = reader.read_chunk_into_coords3(1, &mut out).unwrap();
        assert_eq!(read, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);

        reader.reset().unwrap();
        let read = reader.read_chunk_into_coords3(1, &mut out).unwrap();
        assert_eq!(read, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn read_chunk_into_coords3_selected_returns_requested_atoms() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("selected_into_no_box.dcd");
        write_test_dcd(
            &path,
            4,
            None,
            &[1.0, 2.0, 3.0, 4.0],
            &[10.0, 20.0, 30.0, 40.0],
            &[100.0, 200.0, 300.0, 400.0],
            false,
        );

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut out = vec![0.0f32; 2 * 3];
        let read = reader
            .read_chunk_into_coords3_selected(1, &[3, 1], &mut out)
            .unwrap();
        assert_eq!(read, 1);
        assert_eq!(out, vec![4.0, 40.0, 400.0, 2.0, 20.0, 200.0]);
    }

    #[test]
    fn skip_frames_advances_dcd_reader_without_decoding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("skip_no_box.dcd");
        write_test_dcd_multi(
            &path,
            1,
            &[
                (None, vec![1.0], vec![2.0], vec![3.0]),
                (None, vec![4.0], vec![5.0], vec![6.0]),
                (None, vec![7.0], vec![8.0], vec![9.0]),
            ],
            false,
        );

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        assert_eq!(reader.skip_frames(2).unwrap(), 2);
        let mut out = vec![0.0f32; 3];
        let read = reader.read_chunk_into_coords3(1, &mut out).unwrap();
        assert_eq!(read, 1);
        assert_eq!(out, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn read_frame_with_zero_unitcell_treated_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_box.dcd");
        write_test_dcd(
            &path,
            1,
            Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            &[1.0],
            &[2.0],
            &[3.0],
            true,
        );

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        let frames = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(frames, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.box_[0], Box3::None);
        assert_eq!(chunk.coords[0], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn read_frame_with_charmm_unitcell_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("charmm_box.dcd");
        // CHARMM/OpenMM unitcell order: [A, gamma, B, beta, alpha, C].
        // gamma/beta/alpha are stored as cosines in this case.
        write_test_dcd_with_charmm_version(
            &path,
            1,
            Some([10.0, 0.0, 20.0, 0.0, 0.0, 30.0]),
            &[1.0],
            &[2.0],
            &[3.0],
            true,
            24,
        );

        let mut reader = DcdReader::open(&path, 1.0).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        let frames = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(frames, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0
            }
        );
    }

    fn write_test_dcd(
        path: &Path,
        n_atoms: usize,
        unitcell: Option<[f64; 6]>,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        set_unitcell_flag: bool,
    ) {
        write_test_dcd_with_charmm_version(path, n_atoms, unitcell, x, y, z, set_unitcell_flag, 0);
    }

    fn write_test_dcd_with_charmm_version(
        path: &Path,
        n_atoms: usize,
        unitcell: Option<[f64; 6]>,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        set_unitcell_flag: bool,
        charmm_version: i32,
    ) {
        assert_eq!(x.len(), n_atoms);
        assert_eq!(y.len(), n_atoms);
        assert_eq!(z.len(), n_atoms);

        let mut file = File::create(path).unwrap();

        let mut header = Vec::with_capacity(84);
        header.extend_from_slice(b"CORD");
        let mut icntrl = [0i32; 20];
        icntrl[0] = 1;
        if set_unitcell_flag {
            icntrl[10] = 1;
        }
        icntrl[19] = charmm_version;
        for value in icntrl {
            header.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(header.len(), 84);
        write_record(&mut file, &header);

        let mut title = Vec::with_capacity(84);
        title.extend_from_slice(&1i32.to_le_bytes());
        let mut line = [b' '; 80];
        let text = b"WARP_MD_TEST_DCD";
        line[..text.len()].copy_from_slice(text);
        title.extend_from_slice(&line);
        write_record(&mut file, &title);

        write_record(&mut file, &(n_atoms as i32).to_le_bytes());

        if let Some(cell) = unitcell {
            let mut payload = Vec::with_capacity(48);
            for value in cell {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            write_record(&mut file, &payload);
        }

        write_f32_record(&mut file, x);
        write_f32_record(&mut file, y);
        write_f32_record(&mut file, z);
    }

    fn write_test_dcd_multi(
        path: &Path,
        n_atoms: usize,
        frames: &[(Option<[f64; 6]>, Vec<f32>, Vec<f32>, Vec<f32>)],
        set_unitcell_flag: bool,
    ) {
        let mut file = File::create(path).unwrap();

        let mut header = Vec::with_capacity(84);
        header.extend_from_slice(b"CORD");
        let mut icntrl = [0i32; 20];
        icntrl[0] = frames.len() as i32;
        if set_unitcell_flag {
            icntrl[10] = 1;
        }
        for value in icntrl {
            header.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(header.len(), 84);
        write_record(&mut file, &header);

        let mut title = Vec::with_capacity(84);
        title.extend_from_slice(&1i32.to_le_bytes());
        let mut line = [b' '; 80];
        let text = b"WARP_MD_TEST_DCD";
        line[..text.len()].copy_from_slice(text);
        title.extend_from_slice(&line);
        write_record(&mut file, &title);

        write_record(&mut file, &(n_atoms as i32).to_le_bytes());

        for (unitcell, x, y, z) in frames {
            assert_eq!(x.len(), n_atoms);
            assert_eq!(y.len(), n_atoms);
            assert_eq!(z.len(), n_atoms);
            if let Some(cell) = unitcell {
                let mut payload = Vec::with_capacity(48);
                for value in cell {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
                write_record(&mut file, &payload);
            }
            write_f32_record(&mut file, x);
            write_f32_record(&mut file, y);
            write_f32_record(&mut file, z);
        }
    }

    fn write_record(file: &mut File, payload: &[u8]) {
        let len = u32::try_from(payload.len()).unwrap();
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(payload).unwrap();
        file.write_all(&len.to_le_bytes()).unwrap();
    }

    fn write_f32_record(file: &mut File, values: &[f32]) {
        let mut payload = Vec::with_capacity(values.len() * 4);
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        write_record(file, &payload);
    }

    #[test]
    fn write_dcd_simple() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = DcdWriter::create(path, 2, 1).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 20.0,
                    lz: 30.0,
                },
            )
            .unwrap();
        writer.flush().unwrap();

        let mut reader = DcdReader::open(path, 1.0).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 1);
        builder.set_requirements(true, false);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish_take().unwrap();
        assert_eq!(chunk.coords[0][0], 1.0);
        assert_eq!(chunk.coords[0][1], 2.0);
        assert_eq!(chunk.coords[0][2], 3.0);
        assert_eq!(chunk.coords[1][0], 4.0);
        assert_eq!(chunk.coords[1][1], 5.0);
        assert_eq!(chunk.coords[1][2], 6.0);
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0,
            }
        );
    }

    #[test]
    fn dcd_unitcell_payload_uses_namd_vmd_cosine_convention() {
        let gamma = 60.0f32;
        let beta = 70.0f32;
        let alpha = 80.0f32;
        let a = 10.0f32;
        let b = 20.0f32;
        let c = 30.0f32;
        let gamma_rad = gamma.to_radians();
        let beta_rad = beta.to_radians();
        let alpha_rad = alpha.to_radians();
        let ax = a;
        let bx = b * gamma_rad.cos();
        let by = b * gamma_rad.sin();
        let cx = c * beta_rad.cos();
        let cy = c * (alpha_rad.cos() - beta_rad.cos() * gamma_rad.cos()) / gamma_rad.sin();
        let cz = (c * c - cx * cx - cy * cy).sqrt();
        let payload = box_to_dcd_payload(Box3::Triclinic {
            m: [ax, 0.0, 0.0, bx, by, 0.0, cx, cy, cz],
        });
        let values = std::array::from_fn::<_, 6, _>(|i| {
            let start = i * 8;
            f64::from_le_bytes(payload[start..start + 8].try_into().unwrap())
        });
        assert!((values[0] - a as f64).abs() < 1e-6);
        assert!((values[1] - gamma_rad.cos() as f64).abs() < 1e-6);
        assert!((values[2] - b as f64).abs() < 1e-6);
        assert!((values[3] - beta_rad.cos() as f64).abs() < 1e-6);
        assert!((values[4] - alpha_rad.cos() as f64).abs() < 1e-6);
        assert!((values[5] - c as f64).abs() < 1e-6);
    }
}
