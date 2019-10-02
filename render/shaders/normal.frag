#version 330

in vec4 vertex;
in vec4 normal;

out vec4 out_color;

void main() {
  out_color = normal;
  out_color.a = 0.5;
}
