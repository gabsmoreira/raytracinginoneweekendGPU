#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include <chrono>
#include <unistd.h>

__device__ vec3 color(const ray& r, hitable *world) {
    hit_record rec;
    if (world->hit(r, 0.0, MAXFLOAT, rec)) {
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void kernel_function(float *pixels, int lenX, int lenY, hitable **world){
    // definindo os indices com base no id do bloco e da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index;
    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    // se os indices forem maiores que imagens, a funcao nao retorna nada
    if((i >= lenX) || (j >= lenY)) return;

    // calculo do ray tracing
    float u = float(i) / float(lenX);
    float v = float(j) / float(lenY);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    vec3 col = color(r, *world);
    
    // definindo indice do vetor da GPU para iterar como se fosse uma matriz
    index = i*3 + j*lenX*3;

    // preenchendo pixels
    pixels[index + 0] = col[0];
    pixels[index + 1] = col[1];
    pixels[index + 2] = col[2];
}

__global__ void kernel_init(hitable **list, hitable **world){

    // adiciona esferas na lista de hitables
    *(list) = new sphere(vec3(0,0,-1), 0.5);
    *(list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *(list+2) = new sphere(vec3(1, 0,-1), 0.5);
    *(list+3) = new sphere(vec3(-1, 0,-1), 0.5);

    // cria world com a lista de hitables
    *world = new hitable_list(list, 3);

}

int main() {
    // comeca a contar o tempo
    auto start = std::chrono::steady_clock::now();

    // tamanho da imagem (480 x 320)
    int nx = 480;
    int ny = 320;

    // definindo tamanho das variaveis
    int num_pixels = nx * ny;
    size_t size_pixels = 3 * num_pixels*sizeof(float);
    size_t size_list = 4 * sizeof(hitable *);
    size_t size_world = sizeof(hitable *);
    
    // cria pixels na memoria da GPU
    float *pixels;
    cudaMalloc((void **)&pixels, size_pixels);

    // cria pixels na memoria da CPU
    float *pixelsCPU;
    pixelsCPU = (float *)malloc(size_pixels);

    // cria list e word na memoria da GPU
    hitable **list;
    hitable **world;
    cudaMalloc((void **)&list, size_list);
    cudaMalloc((void **)&world, size_world);

    // definindo o tamanho do bloco
    dim3 blocks(nx/8+1,ny/8+1);

    // definindo threads
    dim3 threads(8,8);

    // chamando o kernel init para criar hitable list e world
    kernel_init<<<1, 1>>>(list, world);

    // sincronizar kernels
    cudaDeviceSynchronize();
    cudaGetLastError();

    // chamando a funcao que calcula os pixels
    kernel_function<<<blocks, threads>>>(pixels, nx, ny, world);

    // memcopy do pixel device -> host
    cudaMemcpy(pixelsCPU, pixels, size_pixels, cudaMemcpyDeviceToHost);

    cudaGetLastError();
    // copia valores do pixelCPU para imagem
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            int index = j*3*nx + i*3;
            float r = pixelsCPU[index + 0];
            float g = pixelsCPU[index + 1];
            float b = pixelsCPU[index + 2];
            int ir = int(255.99*r); 
            int ig = int(255.99*g); 
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    // medicao de tempo
    auto end = std::chrono::steady_clock::now();
    std::cerr << "Elapsed time in seconds : " 
		<< std::chrono::duration<double>(end - start).count()
		<< " s" << std::endl;

}
