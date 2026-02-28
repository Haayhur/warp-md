use super::*;

impl GpuContext {
    pub fn hbond_counts_angle(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        donors: &GpuSelection,
        hydrogens: &GpuSelection,
        acceptors: &GpuSelection,
        dist2: f32,
        cos_cutoff: f32,
    ) -> TrajResult<Vec<u32>> {
        if n_frames == 0 || donors.n_sel == 0 || acceptors.n_sel == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut counts = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;
        let n_donors = donors.n_sel as i32;
        let n_acceptors = acceptors.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let total_pairs = donors.n_sel * acceptors.n_sel;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total_pairs.max(1), block), n_frames_i32 as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.hbond_count_angle);
            builder.arg(&coords_dev);
            builder.arg(&donors.sel);
            builder.arg(&hydrogens.sel);
            builder.arg(&acceptors.sel);
            builder.arg(&n_donors);
            builder.arg(&n_acceptors);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&dist2);
            builder.arg(&cos_cutoff);
            builder.arg(&mut counts);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0u32; n_frames];
        stream
            .memcpy_dtoh(&counts, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn msd_time_lag(
        &self,
        com: &[f32],
        times: &[f32],
        type_ids: &[u32],
        type_counts: &[u32],
        n_groups: usize,
        n_types: usize,
        ndframe: usize,
        axis: Option<[f32; 3]>,
        frame_decimation: Option<(usize, usize)>,
        dt_decimation: Option<(usize, usize, usize, usize)>,
        time_binning: (f32, f32),
    ) -> TrajResult<(Vec<f32>, Vec<u32>)> {
        let n_frames = times.len();
        if n_frames < 2 || n_groups == 0 || ndframe == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let components = if axis.is_some() { 5 } else { 4 };
        let cols = components * (n_types + 1);
        let stream = &self.inner.stream;
        let com_dev = stream.clone_htod(com).map_err(map_driver_err)?;
        let times_dev = stream.clone_htod(times).map_err(map_driver_err)?;
        let type_ids_dev = stream.clone_htod(type_ids).map_err(map_driver_err)?;
        let type_counts_dev = stream.clone_htod(type_counts).map_err(map_driver_err)?;
        let mut out_dev = stream
            .alloc_zeros::<f32>((ndframe + 1) * cols)
            .map_err(map_driver_err)?;
        let mut n_diff_dev = stream
            .alloc_zeros::<u32>(ndframe + 1)
            .map_err(map_driver_err)?;

        let [ax, ay, az] = axis.unwrap_or([0.0, 0.0, 0.0]);
        let axis_enabled = if axis.is_some() { 1i32 } else { 0i32 };
        let (frame_dec_start, frame_dec_stride, frame_dec_enabled) = match frame_decimation {
            Some((start, stride)) => (start as i32, stride as i32, 1i32),
            None => (0i32, 1i32, 0i32),
        };
        let (dt_cut1, dt_stride1, dt_cut2, dt_stride2, dt_dec_enabled) = match dt_decimation {
            Some((cut1, stride1, cut2, stride2)) => (
                cut1 as i32,
                stride1 as i32,
                cut2 as i32,
                stride2 as i32,
                1i32,
            ),
            None => (0i32, 1i32, 0i32, 1i32, 0i32),
        };
        let dt0 = times[1] - times[0];
        let n_groups_i32 = n_groups as i32;
        let n_types_i32 = n_types as i32;
        let n_frames_i32 = n_frames as i32;
        let ndframe_i32 = ndframe as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, n_groups as u32),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.msd_time_lag);
            builder.arg(&com_dev);
            builder.arg(&times_dev);
            builder.arg(&type_ids_dev);
            builder.arg(&type_counts_dev);
            builder.arg(&n_groups_i32);
            builder.arg(&n_types_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&ndframe_i32);
            builder.arg(&dt0);
            builder.arg(&time_binning.0);
            builder.arg(&time_binning.1);
            builder.arg(&frame_dec_start);
            builder.arg(&frame_dec_stride);
            builder.arg(&frame_dec_enabled);
            builder.arg(&dt_cut1);
            builder.arg(&dt_stride1);
            builder.arg(&dt_cut2);
            builder.arg(&dt_stride2);
            builder.arg(&dt_dec_enabled);
            builder.arg(&axis_enabled);
            builder.arg(&ax);
            builder.arg(&ay);
            builder.arg(&az);
            builder.arg(&mut out_dev);
            builder.arg(&mut n_diff_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut out = vec![0.0f32; (ndframe + 1) * cols];
        let mut n_diff = vec![0u32; ndframe + 1];
        stream
            .memcpy_dtoh(&out_dev, &mut out)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&n_diff_dev, &mut n_diff)
            .map_err(map_driver_err)?;
        Ok((out, n_diff))
    }

    pub fn rotacf_time_lag(
        &self,
        orient: &[f32],
        times: &[f32],
        type_ids: &[u32],
        type_counts: &[u32],
        n_groups: usize,
        n_types: usize,
        ndframe: usize,
        frame_decimation: Option<(usize, usize)>,
        dt_decimation: Option<(usize, usize, usize, usize)>,
        time_binning: (f32, f32),
    ) -> TrajResult<(Vec<f32>, Vec<f32>, Vec<u32>)> {
        let n_frames = times.len();
        if n_frames < 2 || n_groups == 0 || ndframe == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let stride = n_types + 1;
        let stream = &self.inner.stream;
        let orient_dev = stream.clone_htod(orient).map_err(map_driver_err)?;
        let times_dev = stream.clone_htod(times).map_err(map_driver_err)?;
        let type_ids_dev = stream.clone_htod(type_ids).map_err(map_driver_err)?;
        let type_counts_dev = stream.clone_htod(type_counts).map_err(map_driver_err)?;
        let mut corr_dev = stream
            .alloc_zeros::<f32>((ndframe + 1) * stride)
            .map_err(map_driver_err)?;
        let mut corr_p2_dev = stream
            .alloc_zeros::<f32>((ndframe + 1) * stride)
            .map_err(map_driver_err)?;
        let mut n_diff_dev = stream
            .alloc_zeros::<u32>(ndframe + 1)
            .map_err(map_driver_err)?;

        let (frame_dec_start, frame_dec_stride, frame_dec_enabled) = match frame_decimation {
            Some((start, stride)) => (start as i32, stride as i32, 1i32),
            None => (0i32, 1i32, 0i32),
        };
        let (dt_cut1, dt_stride1, dt_cut2, dt_stride2, dt_dec_enabled) = match dt_decimation {
            Some((cut1, stride1, cut2, stride2)) => (
                cut1 as i32,
                stride1 as i32,
                cut2 as i32,
                stride2 as i32,
                1i32,
            ),
            None => (0i32, 1i32, 0i32, 1i32, 0i32),
        };
        let dt0 = times[1] - times[0];
        let n_groups_i32 = n_groups as i32;
        let n_types_i32 = n_types as i32;
        let n_frames_i32 = n_frames as i32;
        let ndframe_i32 = ndframe as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, n_groups as u32),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rotacf_time_lag);
            builder.arg(&orient_dev);
            builder.arg(&times_dev);
            builder.arg(&type_ids_dev);
            builder.arg(&type_counts_dev);
            builder.arg(&n_groups_i32);
            builder.arg(&n_types_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&ndframe_i32);
            builder.arg(&dt0);
            builder.arg(&time_binning.0);
            builder.arg(&time_binning.1);
            builder.arg(&frame_dec_start);
            builder.arg(&frame_dec_stride);
            builder.arg(&frame_dec_enabled);
            builder.arg(&dt_cut1);
            builder.arg(&dt_stride1);
            builder.arg(&dt_cut2);
            builder.arg(&dt_stride2);
            builder.arg(&dt_dec_enabled);
            builder.arg(&mut corr_dev);
            builder.arg(&mut corr_p2_dev);
            builder.arg(&mut n_diff_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut corr = vec![0.0f32; (ndframe + 1) * stride];
        let mut corr_p2 = vec![0.0f32; (ndframe + 1) * stride];
        let mut n_diff = vec![0u32; ndframe + 1];
        stream
            .memcpy_dtoh(&corr_dev, &mut corr)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&corr_p2_dev, &mut corr_p2)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&n_diff_dev, &mut n_diff)
            .map_err(map_driver_err)?;
        Ok((corr, corr_p2, n_diff))
    }

    pub fn xcorr_time_lag(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        n_frames: usize,
        ndframe: usize,
    ) -> TrajResult<(Vec<f32>, Vec<u32>)> {
        if n_frames < 2 || ndframe == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if vec_a.len() < n_frames * 3 || vec_b.len() < n_frames * 3 {
            return Err(TrajError::Mismatch("xcorr input length mismatch".into()));
        }
        let stream = &self.inner.stream;
        let vec_a_dev = stream.clone_htod(vec_a).map_err(map_driver_err)?;
        let vec_b_dev = stream.clone_htod(vec_b).map_err(map_driver_err)?;
        let mut out_dev = stream
            .alloc_zeros::<f32>(ndframe + 1)
            .map_err(map_driver_err)?;
        let mut n_diff_dev = stream
            .alloc_zeros::<u32>(ndframe + 1)
            .map_err(map_driver_err)?;

        let n_frames_i32 = n_frames as i32;
        let ndframe_i32 = ndframe as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.xcorr_time_lag);
            builder.arg(&vec_a_dev);
            builder.arg(&vec_b_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&ndframe_i32);
            builder.arg(&mut out_dev);
            builder.arg(&mut n_diff_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut out = vec![0.0f32; ndframe + 1];
        let mut n_diff = vec![0u32; ndframe + 1];
        stream
            .memcpy_dtoh(&out_dev, &mut out)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&n_diff_dev, &mut n_diff)
            .map_err(map_driver_err)?;
        Ok((out, n_diff))
    }

    pub fn timecorr_series_lag(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        n_frames: usize,
        n_items: usize,
        ndframe: usize,
    ) -> TrajResult<(Vec<f32>, Vec<u32>)> {
        if n_frames < 2 || ndframe == 0 || n_items == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let expected = n_frames.saturating_mul(n_items).saturating_mul(3);
        if vec_a.len() < expected || vec_b.len() < expected {
            return Err(TrajError::Mismatch("timecorr input length mismatch".into()));
        }
        let stream = &self.inner.stream;
        let vec_a_dev = stream.clone_htod(vec_a).map_err(map_driver_err)?;
        let vec_b_dev = stream.clone_htod(vec_b).map_err(map_driver_err)?;
        let mut out_dev = stream
            .alloc_zeros::<f32>(ndframe + 1)
            .map_err(map_driver_err)?;
        let mut n_diff_dev = stream
            .alloc_zeros::<u32>(ndframe + 1)
            .map_err(map_driver_err)?;

        let n_frames_i32 = n_frames as i32;
        let n_items_i32 = n_items as i32;
        let ndframe_i32 = ndframe as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, n_items as u32),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.timecorr_series_lag);
            builder.arg(&vec_a_dev);
            builder.arg(&vec_b_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&n_items_i32);
            builder.arg(&ndframe_i32);
            builder.arg(&mut out_dev);
            builder.arg(&mut n_diff_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut out = vec![0.0f32; ndframe + 1];
        let mut n_diff = vec![0u32; ndframe + 1];
        stream
            .memcpy_dtoh(&out_dev, &mut out)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&n_diff_dev, &mut n_diff)
            .map_err(map_driver_err)?;
        Ok((out, n_diff))
    }

    pub fn conductivity_time_lag(
        &self,
        com: &[f32],
        times: &[f32],
        charges: &[f32],
        type_ids: &[u32],
        type_charge: &[f32],
        n_groups: usize,
        n_types: usize,
        ndframe: usize,
        transference: bool,
        frame_decimation: Option<(usize, usize)>,
        dt_decimation: Option<(usize, usize, usize, usize)>,
        time_binning: (f32, f32),
    ) -> TrajResult<(Vec<f32>, Vec<u32>, usize)> {
        let n_frames = times.len();
        if n_frames < 2 || n_groups == 0 || ndframe == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let cols = if transference {
            n_types * n_types + 1
        } else {
            1
        };
        let stream = &self.inner.stream;
        let com_dev = stream.clone_htod(com).map_err(map_driver_err)?;
        let times_dev = stream.clone_htod(times).map_err(map_driver_err)?;
        let charges_dev = stream.clone_htod(charges).map_err(map_driver_err)?;
        let type_ids_dev = stream.clone_htod(type_ids).map_err(map_driver_err)?;
        let type_charge_dev = stream.clone_htod(type_charge).map_err(map_driver_err)?;
        let mut out_dev = stream
            .alloc_zeros::<f32>((ndframe + 1) * cols)
            .map_err(map_driver_err)?;
        let mut n_diff_dev = stream
            .alloc_zeros::<u32>(ndframe + 1)
            .map_err(map_driver_err)?;

        let (frame_dec_start, frame_dec_stride, frame_dec_enabled) = match frame_decimation {
            Some((start, stride)) => (start as i32, stride as i32, 1i32),
            None => (0i32, 1i32, 0i32),
        };
        let (dt_cut1, dt_stride1, dt_cut2, dt_stride2, dt_dec_enabled) = match dt_decimation {
            Some((cut1, stride1, cut2, stride2)) => (
                cut1 as i32,
                stride1 as i32,
                cut2 as i32,
                stride2 as i32,
                1i32,
            ),
            None => (0i32, 1i32, 0i32, 1i32, 0i32),
        };
        let dt0 = times[1] - times[0];
        let n_groups_i32 = n_groups as i32;
        let n_types_i32 = n_types as i32;
        let n_frames_i32 = n_frames as i32;
        let ndframe_i32 = ndframe as i32;
        let cols_i32 = cols as i32;

        let block = 256u32;
        let cfg_total = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.conductivity_total);
            builder.arg(&com_dev);
            builder.arg(&times_dev);
            builder.arg(&charges_dev);
            builder.arg(&n_groups_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&ndframe_i32);
            builder.arg(&dt0);
            builder.arg(&time_binning.0);
            builder.arg(&time_binning.1);
            builder.arg(&frame_dec_start);
            builder.arg(&frame_dec_stride);
            builder.arg(&frame_dec_enabled);
            builder.arg(&dt_cut1);
            builder.arg(&dt_stride1);
            builder.arg(&dt_cut2);
            builder.arg(&dt_stride2);
            builder.arg(&dt_dec_enabled);
            builder.arg(&cols_i32);
            builder.arg(&mut out_dev);
            builder.arg(&mut n_diff_dev);
            builder.launch(cfg_total).map_err(map_driver_err)?;
        }

        if transference {
            let total_pairs = n_groups * n_groups;
            let cfg = LaunchConfig {
                block_dim: (block, 1, 1),
                grid_dim: (
                    ceil_div(total_pairs.max(1), block),
                    n_frames as u32,
                    n_frames as u32,
                ),
                shared_mem_bytes: 0,
            };
            unsafe {
                let mut builder =
                    stream.launch_builder(&self.inner.kernels.conductivity_transference);
                builder.arg(&com_dev);
                builder.arg(&times_dev);
                builder.arg(&charges_dev);
                builder.arg(&type_ids_dev);
                builder.arg(&type_charge_dev);
                builder.arg(&n_groups_i32);
                builder.arg(&n_types_i32);
                builder.arg(&n_frames_i32);
                builder.arg(&ndframe_i32);
                builder.arg(&dt0);
                builder.arg(&time_binning.0);
                builder.arg(&time_binning.1);
                builder.arg(&frame_dec_start);
                builder.arg(&frame_dec_stride);
                builder.arg(&frame_dec_enabled);
                builder.arg(&dt_cut1);
                builder.arg(&dt_stride1);
                builder.arg(&dt_cut2);
                builder.arg(&dt_stride2);
                builder.arg(&dt_dec_enabled);
                builder.arg(&cols_i32);
                builder.arg(&mut out_dev);
                builder.launch(cfg).map_err(map_driver_err)?;
            }
        }

        let mut out = vec![0.0f32; (ndframe + 1) * cols];
        let mut n_diff = vec![0u32; ndframe + 1];
        stream
            .memcpy_dtoh(&out_dev, &mut out)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&n_diff_dev, &mut n_diff)
            .map_err(map_driver_err)?;
        Ok((out, n_diff, cols))
    }
}
