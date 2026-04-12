use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let vendor = PathBuf::from("vendor/tng_io");
    println!("cargo:rerun-if-changed=build.rs");
    rerun_tree(&vendor);

    let include = vendor.join("include");
    let zlib = vendor.join("external/zlib");

    let mut build = cc::Build::new();
    build
        .include(&include)
        .include(&zlib)
        .warnings(false)
        .define("USE_STD_INTTYPES_H", None);

    if std::env::var("CARGO_CFG_TARGET_ENDIAN").ok().as_deref() == Some("big") {
        build.define("TNG_INTEGER_BIG_ENDIAN", None);
    }

    for rel in [
        "src/lib/tng_io.c",
        "src/lib/md5.c",
        "src/compression/bwlzh.c",
        "src/compression/bwt.c",
        "src/compression/coder.c",
        "src/compression/dict.c",
        "src/compression/fixpoint.c",
        "src/compression/huffman.c",
        "src/compression/huffmem.c",
        "src/compression/lz77.c",
        "src/compression/merge_sort.c",
        "src/compression/mtf.c",
        "src/compression/rle.c",
        "src/compression/tng_compress.c",
        "src/compression/vals16.c",
        "src/compression/warnmalloc.c",
        "src/compression/widemuldiv.c",
        "src/compression/xtc2.c",
        "src/compression/xtc3.c",
        "external/zlib/adler32.c",
        "external/zlib/compress.c",
        "external/zlib/crc32.c",
        "external/zlib/deflate.c",
        "external/zlib/inffast.c",
        "external/zlib/inflate.c",
        "external/zlib/inftrees.c",
        "external/zlib/trees.c",
        "external/zlib/uncompr.c",
        "external/zlib/zutil.c",
    ] {
        build.file(vendor.join(rel));
    }

    build.compile("traj_io_tng");

    if std::env::var("CARGO_CFG_TARGET_FAMILY").ok().as_deref() == Some("unix") {
        println!("cargo:rustc-link-lib=m");
    }
}

fn rerun_tree(path: &Path) {
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            println!("cargo:rerun-if-changed={}", path.display());
            if path.is_dir() {
                rerun_tree(&path);
            }
        }
    }
}
