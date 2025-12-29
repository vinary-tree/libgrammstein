//! GPU-accelerated operations for embedding training and similarity search.
//!
//! This module provides wgpu-based acceleration for compute-intensive
//! embedding operations:
//!
//! - Batch dot product computation
//! - Sigmoid activation
//! - Gradient accumulation
//! - Matrix-vector multiplication for similarity search
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::embedding::gpu::{GpuContext, GpuBatchDotProduct};
//!
//! // Initialize GPU context
//! let context = GpuContext::new()?;
//!
//! // Create batch dot product processor
//! let dot_product = GpuBatchDotProduct::new(&context, 128)?; // dim=128
//!
//! // Compute dot products
//! let results = dot_product.compute(&vectors_a, &vectors_b)?;
//! ```

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Failed to get GPU device.
    NoDevice(wgpu::RequestDeviceError),
    /// Shader compilation error.
    ShaderError(String),
    /// Buffer operation error.
    BufferError(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// GPU operation timeout.
    Timeout,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "No suitable GPU adapter found"),
            Self::NoDevice(e) => write!(f, "Failed to get GPU device: {}", e),
            Self::ShaderError(msg) => write!(f, "Shader error: {}", msg),
            Self::BufferError(msg) => write!(f, "Buffer error: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::Timeout => write!(f, "GPU operation timeout"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;

/// Uniform data for shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    /// Embedding dimension.
    dim: u32,
    /// Number of vectors in batch.
    batch_size: u32,
    /// Learning rate (for gradient ops).
    learning_rate: f32,
    /// Padding for 16-byte alignment.
    _padding: u32,
}

/// GPU context holding device and queue.
pub struct GpuContext {
    /// wgpu device.
    device: Arc<wgpu::Device>,
    /// wgpu queue.
    queue: Arc<wgpu::Queue>,
    /// Adapter info for logging.
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new GPU context.
    ///
    /// Attempts to find a high-performance GPU adapter.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Create a new GPU context asynchronously.
    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Using GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("grammstein_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .map_err(GpuError::NoDevice)?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Get the GPU adapter name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get the GPU backend.
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    /// Get reference to device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// WGSL shader for batch dot products.
const DOT_PRODUCT_SHADER: &str = r#"
struct Uniforms {
    dim: u32,
    batch_size: u32,
    learning_rate: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vectors_a: array<f32>;
@group(0) @binding(2) var<storage, read> vectors_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.batch_size) {
        return;
    }

    var sum: f32 = 0.0;
    let offset = idx * uniforms.dim;

    for (var i: u32 = 0u; i < uniforms.dim; i = i + 1u) {
        sum = sum + vectors_a[offset + i] * vectors_b[offset + i];
    }

    results[idx] = sum;
}
"#;

/// WGSL shader for sigmoid activation.
const SIGMOID_SHADER: &str = r#"
struct Uniforms {
    dim: u32,
    batch_size: u32,
    learning_rate: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.batch_size) {
        return;
    }

    output[idx] = sigmoid(input[idx]);
}
"#;

/// WGSL shader for gradient accumulation (axpy: y += alpha * x).
const GRADIENT_ACCUM_SHADER: &str = r#"
struct Uniforms {
    dim: u32,
    batch_size: u32,
    learning_rate: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> embeddings: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = uniforms.batch_size * uniforms.dim;
    if (idx >= total_elements) {
        return;
    }

    embeddings[idx] = embeddings[idx] + uniforms.learning_rate * gradients[idx];
}
"#;

/// WGSL shader for matrix-vector multiplication (for similarity search).
const MATVEC_SHADER: &str = r#"
struct Uniforms {
    dim: u32,
    batch_size: u32,
    learning_rate: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= uniforms.batch_size) {
        return;
    }

    var sum: f32 = 0.0;
    let offset = row * uniforms.dim;

    for (var i: u32 = 0u; i < uniforms.dim; i = i + 1u) {
        sum = sum + matrix[offset + i] * vector[i];
    }

    result[row] = sum;
}
"#;

/// GPU-accelerated batch dot product computation.
pub struct GpuBatchDotProduct {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dim: usize,
}

impl GpuBatchDotProduct {
    /// Create a new batch dot product processor.
    pub fn new(context: &GpuContext, dim: usize) -> Result<Self> {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dot_product_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DOT_PRODUCT_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_product_bind_group_layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vectors A
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vectors B
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dot_product_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dot_product_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            dim,
        })
    }

    /// Compute batch dot products.
    ///
    /// # Arguments
    ///
    /// * `vectors_a` - First set of vectors (flattened, row-major)
    /// * `vectors_b` - Second set of vectors (flattened, row-major)
    ///
    /// # Returns
    ///
    /// Vector of dot products, one per pair.
    pub fn compute(&self, vectors_a: &[f32], vectors_b: &[f32]) -> Result<Vec<f32>> {
        if vectors_a.len() != vectors_b.len() {
            return Err(GpuError::DimensionMismatch {
                expected: vectors_a.len(),
                got: vectors_b.len(),
            });
        }

        let batch_size = vectors_a.len() / self.dim;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Create buffers
        let uniforms = Uniforms {
            dim: self.dim as u32,
            batch_size: batch_size as u32,
            learning_rate: 0.0,
            _padding: 0,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let vectors_a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vectors_a"),
                contents: bytemuck::cast_slice(vectors_a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let vectors_b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vectors_b"),
                contents: bytemuck::cast_slice(vectors_b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dot_product_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vectors_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: results_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dot_product_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dot_product_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((batch_size as u32 + 255) / 256, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::Timeout)?
            .map_err(|e| GpuError::BufferError(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// GPU-accelerated sigmoid activation.
pub struct GpuSigmoid {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuSigmoid {
    /// Create a new sigmoid processor.
    pub fn new(context: &GpuContext) -> Result<Self> {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sigmoid_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SIGMOID_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sigmoid_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sigmoid_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sigmoid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply sigmoid activation to input values.
    pub fn compute(&self, input: &[f32]) -> Result<Vec<f32>> {
        let batch_size = input.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let uniforms = Uniforms {
            dim: 1,
            batch_size: batch_size as u32,
            learning_rate: 0.0,
            _padding: 0,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sigmoid_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sigmoid_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sigmoid_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((batch_size as u32 + 255) / 256, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::Timeout)?
            .map_err(|e| GpuError::BufferError(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// GPU-accelerated gradient accumulation.
pub struct GpuGradientAccum {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dim: usize,
}

impl GpuGradientAccum {
    /// Create a new gradient accumulator.
    pub fn new(context: &GpuContext, dim: usize) -> Result<Self> {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gradient_accum_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GRADIENT_ACCUM_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gradient_accum_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gradient_accum_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient_accum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            dim,
        })
    }

    /// Apply gradient updates: embeddings += learning_rate * gradients.
    ///
    /// Updates embeddings in-place and returns the updated values.
    pub fn apply(
        &self,
        embeddings: &[f32],
        gradients: &[f32],
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        if embeddings.len() != gradients.len() {
            return Err(GpuError::DimensionMismatch {
                expected: embeddings.len(),
                got: gradients.len(),
            });
        }

        let batch_size = embeddings.len() / self.dim;
        let total_elements = embeddings.len();

        if total_elements == 0 {
            return Ok(Vec::new());
        }

        let uniforms = Uniforms {
            dim: self.dim as u32,
            batch_size: batch_size as u32,
            learning_rate,
            _padding: 0,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let gradients_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gradients"),
                contents: bytemuck::cast_slice(gradients),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let embeddings_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embeddings"),
                contents: bytemuck::cast_slice(embeddings),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (total_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gradient_accum_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gradients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: embeddings_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gradient_accum_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gradient_accum_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((total_elements as u32 + 255) / 256, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &embeddings_buffer,
            0,
            &staging_buffer,
            0,
            (total_elements * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::Timeout)?
            .map_err(|e| GpuError::BufferError(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// GPU-accelerated matrix-vector multiplication for similarity search.
pub struct GpuSimilaritySearch {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dim: usize,
}

impl GpuSimilaritySearch {
    /// Create a new similarity search processor.
    pub fn new(context: &GpuContext, dim: usize) -> Result<Self> {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATVEC_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            dim,
        })
    }

    /// Compute similarities between all matrix rows and a query vector.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Embedding matrix (flattened, row-major)
    /// * `query` - Query vector
    ///
    /// # Returns
    ///
    /// Similarity scores for each row.
    pub fn compute(&self, matrix: &[f32], query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let num_rows = matrix.len() / self.dim;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let uniforms = Uniforms {
            dim: self.dim as u32,
            batch_size: num_rows as u32,
            learning_rate: 0.0,
            _padding: 0,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let matrix_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix"),
                contents: bytemuck::cast_slice(matrix),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("query"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size: (num_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (num_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matvec_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((num_rows as u32 + 255) / 256, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            (num_rows * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::Timeout)?
            .map_err(|e| GpuError::BufferError(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// Convenience wrapper for all GPU operations.
pub struct GpuAccelerator {
    context: GpuContext,
    dot_product: GpuBatchDotProduct,
    sigmoid: GpuSigmoid,
    gradient_accum: GpuGradientAccum,
    similarity_search: GpuSimilaritySearch,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator with given embedding dimension.
    pub fn new(dim: usize) -> Result<Self> {
        let context = GpuContext::new()?;
        let dot_product = GpuBatchDotProduct::new(&context, dim)?;
        let sigmoid = GpuSigmoid::new(&context)?;
        let gradient_accum = GpuGradientAccum::new(&context, dim)?;
        let similarity_search = GpuSimilaritySearch::new(&context, dim)?;

        Ok(Self {
            context,
            dot_product,
            sigmoid,
            gradient_accum,
            similarity_search,
        })
    }

    /// Get the GPU adapter name.
    pub fn adapter_name(&self) -> &str {
        self.context.adapter_name()
    }

    /// Compute batch dot products.
    pub fn batch_dot_product(&self, vectors_a: &[f32], vectors_b: &[f32]) -> Result<Vec<f32>> {
        self.dot_product.compute(vectors_a, vectors_b)
    }

    /// Apply sigmoid activation.
    pub fn sigmoid(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.sigmoid.compute(input)
    }

    /// Apply gradient updates.
    pub fn apply_gradients(
        &self,
        embeddings: &[f32],
        gradients: &[f32],
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        self.gradient_accum.apply(embeddings, gradients, learning_rate)
    }

    /// Compute similarity scores against a query vector.
    pub fn similarity_search(&self, matrix: &[f32], query: &[f32]) -> Result<Vec<f32>> {
        self.similarity_search.compute(matrix, query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_gpu() -> bool {
        GpuContext::new().is_ok()
    }

    #[test]
    fn test_gpu_context() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let context = GpuContext::new().expect("should create context");
        assert!(!context.adapter_name().is_empty());
    }

    #[test]
    fn test_batch_dot_product() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let context = GpuContext::new().unwrap();
        let dot_product = GpuBatchDotProduct::new(&context, 4).unwrap();

        // 2 pairs of 4D vectors
        let vectors_a = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let vectors_b = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

        let results = dot_product.compute(&vectors_a, &vectors_b).unwrap();

        assert_eq!(results.len(), 2);
        // First pair: 1*1 + 2*1 + 3*1 + 4*1 = 10
        assert!((results[0] - 10.0).abs() < 1e-5);
        // Second pair: 0.5*2 + 0.5*2 + 0.5*2 + 0.5*2 = 4
        assert!((results[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let context = GpuContext::new().unwrap();
        let sigmoid = GpuSigmoid::new(&context).unwrap();

        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let results = sigmoid.compute(&input).unwrap();

        assert_eq!(results.len(), 5);
        // sigmoid(0) = 0.5
        assert!((results[0] - 0.5).abs() < 1e-5);
        // sigmoid(1) ≈ 0.731
        assert!((results[1] - 0.7310586).abs() < 1e-4);
        // sigmoid(-1) ≈ 0.269
        assert!((results[2] - 0.2689414).abs() < 1e-4);
    }

    #[test]
    fn test_gradient_accumulation() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let context = GpuContext::new().unwrap();
        let grad_accum = GpuGradientAccum::new(&context, 4).unwrap();

        // 2 embeddings of dim 4
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gradients = vec![0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2];
        let learning_rate = 1.0;

        let results = grad_accum
            .apply(&embeddings, &gradients, learning_rate)
            .unwrap();

        assert_eq!(results.len(), 8);
        // First embedding: 1 + 1*0.1 = 1.1
        assert!((results[0] - 1.1).abs() < 1e-5);
        // Last embedding: 8 + 1*0.2 = 8.2
        assert!((results[7] - 8.2).abs() < 1e-5);
    }

    #[test]
    fn test_similarity_search() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let context = GpuContext::new().unwrap();
        let search = GpuSimilaritySearch::new(&context, 4).unwrap();

        // 3 embeddings of dim 4
        let matrix = vec![
            1.0, 0.0, 0.0, 0.0, // Row 0
            0.0, 1.0, 0.0, 0.0, // Row 1
            0.0, 0.0, 1.0, 0.0, // Row 2
        ];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results = search.compute(&matrix, &query).unwrap();

        assert_eq!(results.len(), 3);
        // Query matches row 0 perfectly
        assert!((results[0] - 1.0).abs() < 1e-5);
        // Query orthogonal to row 1
        assert!((results[1] - 0.0).abs() < 1e-5);
        // Query orthogonal to row 2
        assert!((results[2] - 0.0).abs() < 1e-5);
    }
}
