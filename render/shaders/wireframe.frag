#version 330

uniform vec4 in_color;

in vec4 vertex;
in vec4 normal;

out vec4 color;

void main() {
  color = in_color;
}
