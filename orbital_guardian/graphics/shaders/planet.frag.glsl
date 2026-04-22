#version 330

// Planet fragment shader — Blinn-Phong lighting with fresnel rim glow.
//
// CG concepts demonstrated:
//   - Blinn-Phong reflection model (ambient + diffuse + specular)
//   - Fresnel/rim lighting for atmospheric edge glow
//   - Per-pixel lighting using interpolated world-space normals

uniform vec3 lightDir;       // normalized direction TO the light
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 cameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(cameraPos - vWorldPos);
    vec3 H = normalize(L + V);  // Blinn-Phong half-vector

    // Ambient term
    vec3 ambient = ambientColor * vColor.rgb;

    // Diffuse term (Lambert)
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = lightColor * vColor.rgb * NdotL;

    // Specular term (Blinn-Phong)
    float NdotH = max(dot(N, H), 0.0);
    float spec = pow(NdotH, 32.0);
    vec3 specular = lightColor * spec * 0.3;

    // Fresnel rim glow — simulates atmospheric scattering at the planet edge
    float rim = 1.0 - max(dot(V, N), 0.0);
    rim = pow(rim, 3.0);
    vec3 rimColor = vec3(0.3, 0.5, 1.0);  // soft blue
    float rimIntensity = 0.4;

    vec3 finalColor = ambient + diffuse + specular + rimColor * rim * rimIntensity;
    fragColor = vec4(finalColor, 1.0);
}
