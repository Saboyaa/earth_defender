#version 330

// Impact marker fragment shader — draws a pulsing ring + translucent fill.
// Color is controlled by a uniform so it can be yellow (falling) or red (embedded).

uniform float time;
uniform float urgency;    // 0..1
uniform float opacity;
uniform vec3 markerColor; // e.g. (1, 0.8, 0) for yellow or (1, 0.15, 0.05) for red

out vec4 fragColor;
in vec2 vUV;

void main() {
    vec2 centered = vUV - vec2(0.5);
    float dist = length(centered) * 2.0;

    if (dist > 1.0) discard;

    float pulseSpeed = mix(4.0, 12.0, urgency);
    float pulse = 0.5 + 0.5 * sin(time * pulseSpeed);

    // Outer ring
    float ringOuter = smoothstep(0.75, 0.82, dist) * (1.0 - smoothstep(0.92, 1.0, dist));
    // Inner ring
    float ringInner = smoothstep(0.40, 0.46, dist) * (1.0 - smoothstep(0.50, 0.56, dist));
    // Center fill
    float fill = (1.0 - smoothstep(0.0, 0.8, dist)) * 0.25;

    float ring = max(ringOuter, ringInner * 0.6);
    float alpha = (ring * 0.9 + fill) * pulse * opacity;

    vec3 color = markerColor * (0.7 + 0.3 * ring);

    fragColor = vec4(color, alpha);
}
