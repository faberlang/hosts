// Lit scene shader for the Triga corpus.
//
// Vertex format: position (location 0, float32x3)
//                normal   (location 1, float32x3)
//                color    (location 2, float32x3)
// Stride: 36 bytes (9 f32 per vertex)
//
// Bind group 0:
//   binding 0 — transform storage buffer (64 f32: model 16 + view_proj 16 + padding 32)
//   binding 1 — lighting uniform buffer (12 f32: sun_dir(3)+pad, sun_color(3)+pad,
//               ambient(3)+fog_density)
//
// Fragment path: hemisphere ambient + Lambert sun in linear space, then
// ACES filmic tone mapping, sRGB encode, and exp2 distance fog mixed in
// display space toward the sky clear color.

@group(0) @binding(0) var<storage, read> transform: array<f32>;
@group(0) @binding(1) var<uniform> lighting: LightingUniforms;

struct LightingUniforms {
  sun_direction: vec3<f32>,   // normalized, points toward the light
  _pad0: f32,
  sun_color: vec3<f32>,       // sun intensity (linear, 0..1+)
  _pad1: f32,
  ambient_color: vec3<f32>,   // ambient sky/fill light
  fog_density: f32,  // exp2 fog density; 0 falls back to the shader default
};

struct LitVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
}

struct LitVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective) world_normal: vec3<f32>,
  @location(1) @interpolate(perspective) base_color: vec3<f32>,
  @location(2) @interpolate(perspective) view_depth: f32,
}

@vertex
fn greybox_vertex(input: LitVertexInput) -> LitVertexOutput {
  var out: LitVertexOutput;
  let model = mat4x4<f32>(
    vec4<f32>(transform[0], transform[1], transform[2], transform[3]),
    vec4<f32>(transform[4], transform[5], transform[6], transform[7]),
    vec4<f32>(transform[8], transform[9], transform[10], transform[11]),
    vec4<f32>(transform[12], transform[13], transform[14], transform[15])
  );
  let view_proj = mat4x4<f32>(
    vec4<f32>(transform[16], transform[17], transform[18], transform[19]),
    vec4<f32>(transform[20], transform[21], transform[22], transform[23]),
    vec4<f32>(transform[24], transform[25], transform[26], transform[27]),
    vec4<f32>(transform[28], transform[29], transform[30], transform[31])
  );
  // Normal matrix: for uniform scale, model's upper-left 3x3 suffices.
  let world_pos = model * vec4<f32>(input.position, 1.0);
  out.position = view_proj * world_pos;
  out.world_normal = normalize(input.normal);
  out.base_color = input.color;
  // Clip-space w is the view-space distance in front of the camera; the
  // fragment stage uses it for fog. abs() keeps it robust to projection
  // conventions that flip the sign.
  out.view_depth = abs(out.position.w);
  return out;
}

struct LitFragmentInput {
  @location(0) @interpolate(perspective) world_normal: vec3<f32>,
  @location(1) @interpolate(perspective) base_color: vec3<f32>,
  @location(2) @interpolate(perspective) view_depth: f32,
}

struct LitFragmentOutput {
  @location(0) color: vec4<f32>,
}

// ACES filmic tone mapping (Narkowicz approximation), linear in/out 0..1+.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn greybox_fragment(input: LitFragmentInput) -> LitFragmentOutput {
  var out: LitFragmentOutput;
  let N = normalize(input.world_normal);
  let L = normalize(lighting.sun_direction);
  let NdotL = max(dot(N, L), 0.0);

  // Simple Lambert + ambient with a hemisphere fill term.
  // Up-facing surfaces get sky ambient; down-facing get ground ambient.
  let up_facing = max(N.y, 0.0);
  let down_facing = max(-N.y, 0.0);
  let sky_tint = vec3<f32>(0.4, 0.55, 0.75) * up_facing;
  let ground_tint = vec3<f32>(0.2, 0.18, 0.15) * down_facing;
  let hemi_ambient = lighting.ambient_color + sky_tint * 0.3 + ground_tint * 0.2;

  let diffuse = lighting.sun_color * NdotL;
  // Exposure trim: keeps sun-lit stone from clipping to white after ACES.
  let exposure = 0.75;
  let lit_linear = input.base_color * (hemi_ambient + diffuse) * exposure;

  // Filmic tone map + display encode.
  let lit_display = linear_to_srgb(aces_tonemap(lit_linear));

  // Exp2 distance fog toward the sky clear color, mixed in display space so
  // fully fogged geometry disappears seamlessly into the background.
  let fog_color = vec3<f32>(0.451, 0.620, 0.800);
  let fog_density = select(0.010, lighting.fog_density, lighting.fog_density > 0.0);
  let fog_amount = 1.0 - exp2(-fog_density * fog_density * input.view_depth * input.view_depth * 1.442695);
  let final_color = mix(lit_display, fog_color, clamp(fog_amount, 0.0, 1.0));

  out.color = vec4<f32>(final_color, 1.0);
  return out;
}
