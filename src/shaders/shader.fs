#version 330 core
out vec4 FragColor;


//uniform vec3 cameraPos;
uniform vec4 color;

void main()
{             
    // float ratio = 1.00 / 1.52;
    // vec3 I = normalize(Position - cameraPos);
    // vec3 R = refract(I, normalize(Normal), ratio);
    FragColor = vec4(color);
}