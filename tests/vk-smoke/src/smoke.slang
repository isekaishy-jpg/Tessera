// The smoke test's shaders: one triangle in constant green. `shader.rs` compiles this file with
// the SDK's `slangc` when the program runs; no SPIR-V is committed.

// Vertex position from binding 0, attribute location 0 (`R32G32_SFLOAT`).
[shader("vertex")]
float4 vertexMain([[vk::location(0)]] float2 position) : SV_Position
{
    return float4(position, 0.0, 1.0);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(0.0, 1.0, 0.0, 1.0);
}
