#version 330

// Impact marker vertex shader — transforms a flat disc that sits on the
// planet surface, oriented along the local surface normal.

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 vUV;

void main() {
    vUV = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
