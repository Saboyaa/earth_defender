#version 330

// Atmosphere fragment shader — soft fresnel glow around the planet.
//
// CG concept: the fresnel effect makes surfaces more reflective / glowing
// at grazing angles.  For the atmosphere shell, this creates a soft halo
// that is transparent when viewed head-on and glows at the edges.

uniform vec3 cameraPos;

in vec3 vWorldPos;
in vec3 vNormal;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(cameraPos - vWorldPos);

    // Fresnel: strong at edges, transparent at center
    float fresnel = 1.0 - max(dot(V, N), 0.0);
    fresnel = pow(fresnel, 2.5);

    // Soft blue-cyan atmosphere color
    vec3 atmosColor = vec3(0.3, 0.6, 1.0);
    float alpha = fresnel * 0.6;  // max opacity at the rim

    fragColor = vec4(atmosColor, alpha);
}
