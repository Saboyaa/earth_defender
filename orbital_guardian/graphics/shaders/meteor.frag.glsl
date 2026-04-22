#version 330

// Meteor fragment shader — hot rock with pulsing emissive glow.
//
// CG concepts:
//   - Time-based animation in shaders (pulsing emission)
//   - Emissive term added on top of diffuse lighting
//   - Blinn-Phong diffuse for base rock shading

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform float time;

in vec3 vWorldPos;
in vec3 vNormal;
in vec4 vColor;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightDir);

    // Basic diffuse lighting on the rock surface
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = lightColor * vColor.rgb * NdotL;
    vec3 ambient = ambientColor * vColor.rgb;

    // Pulsing emissive glow — makes the meteor look hot
    float pulse = 0.7 + 0.3 * sin(time * 5.0);
    vec3 emissiveColor = vec3(1.0, 0.4, 0.1);  // orange-red
    vec3 emission = emissiveColor * pulse * 0.5;

    vec3 finalColor = ambient + diffuse + emission;
    fragColor = vec4(finalColor, 1.0);
}
