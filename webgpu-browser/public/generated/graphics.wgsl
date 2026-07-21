struct HelloVoxelVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
}

struct HelloVoxelVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective) color: vec3<f32>,
}

@vertex
fn hello_voxel_vertex(input: HelloVoxelVertexInput) -> HelloVoxelVertexOutput {
  var out: HelloVoxelVertexOutput;
  out.position = vec4<f32>(input.position, 1.0);
  out.color = input.color;
  return out;
}

struct HelloVoxelFragmentInput {
  @location(0) @interpolate(perspective) color: vec3<f32>,
}

struct HelloVoxelFragmentOutput {
  @location(0) color: vec4<f32>,
}

@fragment
fn hello_voxel_fragment(input: HelloVoxelFragmentInput) -> HelloVoxelFragmentOutput {
  var out: HelloVoxelFragmentOutput;
  out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  return out;
}

