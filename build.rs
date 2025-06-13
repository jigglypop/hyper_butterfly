fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::env;
        use bindgen::Builder;
        use cc::Build;
        let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH environment variable not set");
        Build::new()
            .cuda(true)
            .flag("-arch=sm_70")
            .file("src/cuda/kernels.cu")
            .compile("kernels");
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
        println!("cargo:rustc-link-lib=static=cudart");
    }
} 