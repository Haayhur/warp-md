use std::fs;
use traj_engine::feature_store::{FeatureSchema, FeatureStoreReader, FeatureStoreWriter};

#[test]
fn feature_store_roundtrip() {
    let dir = std::env::temp_dir();
    let base = dir.join("warp_md_feature_store_test");
    let bin_path = base.with_extension("bin");
    let json_path = base.with_extension("json");
    let _ = fs::remove_file(&bin_path);
    let _ = fs::remove_file(&json_path);

    let schema = FeatureSchema {
        name: "test".to_string(),
        dtype: "f32".to_string(),
        dim: 3,
        n_streams: 2,
        chunk_frames: 2,
    };
    let mut writer = FeatureStoreWriter::new(&base, schema).expect("writer");
    let data = vec![
        0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ];
    writer.write_chunk(2, &data).expect("write");
    writer.finish().expect("finish");

    let mut reader = FeatureStoreReader::open(&base).expect("open");
    let idx = reader.index();
    assert_eq!(idx.total_frames, 2);
    assert_eq!(idx.schema.dim, 3);
    let chunk = reader.read_chunk(0).expect("read");
    assert_eq!(chunk, data);

    let _ = fs::remove_file(bin_path);
    let _ = fs::remove_file(json_path);
}
